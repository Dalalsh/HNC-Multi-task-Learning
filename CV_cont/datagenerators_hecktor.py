import os, sys
import random
import numpy as np
import cv2
import random
import pdb
import copy

import logging
import pandas as pd
import torchio as tio
import tensorflow as tf
import SimpleITK as sitk

from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import label

import imgaug as ia
from imgaug import augmenters as iaa



def IBSI_resampling(pet, ct, mask, **kwargs):
    # https://github.com/Radiomics/pyradiomics/issues/498
    ibsiLogger = logging.getLogger('radiomics.ibsi')
    # resample image to new spacing, align centers of both resampling grids.
    pixelId = sitk.sitkFloat32
    spacing = kwargs.get('resampledPixelSpacing')
    grayValuePrecision = kwargs.get('grayValuePrecision')

    pet_spacing = np.array(pet.GetSpacing(), dtype='float')
    pet_size = np.array(pet.GetSize(), dtype='float')
    spacing = np.where(np.array(spacing) == 0, pet_spacing, spacing)
    spacingRatio = pet_spacing / spacing
    newSize = np.ceil(pet_size * spacingRatio)
    # Calculate center in real-world coordinates
    pet_center = pet.TransformContinuousIndexToPhysicalPoint((pet_size - 1) / 2)
    new_origin = tuple(np.array(pet.GetOrigin()) + 0.5 * ((pet_size - 1) * pet_spacing - (newSize - 1) * spacing))
    ibsiLogger.info('Resampling from %s to %s (size %s to %s), aligning Centers', pet_spacing, spacing, pet_size, newSize)
    rif = sitk.ResampleImageFilter()
    rif.SetOutputOrigin(new_origin)
    rif.SetSize(np.array(newSize, dtype='int').tolist())
    rif.SetOutputDirection(pet.GetDirection())
    rif.SetOutputSpacing(spacing)
    rif.SetOutputPixelType(pixelId)
    rif.SetInterpolator(sitk.sitkLinear)
    res_pet = rif.Execute(sitk.Cast(pet, pixelId))
    # Round to n decimals (0 = to nearest integer)
    if grayValuePrecision is not None:
        ibsiLogger.debug('Rounding Image Gray values to %d decimals', grayValuePrecision)
        pet_arr = sitk.GetArrayFromImage(res_pet)
        pet_arr = np.round(pet_arr, grayValuePrecision)
        round_pet = sitk.GetImageFromArray(pet_arr)
        round_pet.CopyInformation(res_pet)
        res_pet = round_pet
    # Sanity check: Compare Centers!
    new_center = res_pet.TransformContinuousIndexToPhysicalPoint((newSize - 1) / 2)
    ibsiLogger.debug("diff centers: %s" % np.abs(np.array(pet_center) - np.array(new_center)))
    
    if mask is not None:
        rif.SetOutputPixelType(pixelId)
        rif.SetInterpolator(sitk.sitkNearestNeighbor)
        rif.SetDefaultPixelValue(0)
        res_ma = rif.Execute(sitk.Cast(mask, pixelId))
        res_ma = sitk.BinaryThreshold(res_ma, lowerThreshold=0.5)
        res_ma = sitk.Cast(res_ma, sitk.sitkUInt8)
    else:
        res_ma = None
        
    if ct is not None:
        rif.SetOutputPixelType(pixelId)
        rif.SetInterpolator(sitk.sitkLinear)
        rif.SetDefaultPixelValue(-1000)
        res_ct = rif.Execute(sitk.Cast(ct, pixelId))
        res_ct = sitk.Cast(res_ct, sitk.sitkInt32)
    else:
        res_ct = None
        
    return res_pet, res_ct, res_ma




def registeration_func(moving_image, fixed_image, initializer_state=None):
    
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    registration_method = sitk.ImageRegistrationMethod()
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(moving_image)
    # Fixed bin size
    numberOfHistogramBins = np.ceil(stats_filter.GetMaximum() / 0.1)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(numberOfHistogramBins))
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1, seed=1024)
    # TODO : work on optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=5, minStep=0.001,
                                                                 relaxationFactor=0.5,
                                                                 numberOfIterations=10000,
                                                                 estimateLearningRate=registration_method.EachIteration
                                                                 )
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    if initializer_state is None:
        im_size = np.array(moving_image.GetSize(), dtype='float')
        im_center = moving_image.TransformContinuousIndexToPhysicalPoint((im_size - 1) / 2)
        initial_transform = sitk.VersorRigid3DTransform((0, 0, 0, 1), (0, 0, 0), im_center)
    elif initializer_state == 'center':
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                              sitk.VersorRigid3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    out_transform = registration_method.Execute(fixed_image, moving_image)


    interpolator = sitk.sitkLinear
    default_value = 0.
    moving_resampled = sitk.Resample(moving_image, 
                                     fixed_image, 
                                     out_transform, 
                                     interpolator,
                                     default_value)
    
    return moving_resampled





def hecktor_reampling(pet, ct, mask, resampledPixelSpacing):
    """Retrive from https://github.com/voreille/hecktor/blob/master/src/resampling/resample_2022.py
       Get the bounding boxes of the CT and PT images.
       This works since all images have the same direction
    """
    
    resampledPixelSpacing = [int(_) for _ in resampledPixelSpacing]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampledPixelSpacing)
    
    
    # Get the bounding box of the CT and PET image.
    ct_origin = np.array(ct.GetOrigin())
    pet_origin = np.array(pet.GetOrigin())

    ct_position_max = ct_origin + np.array(ct.GetSize()) * np.array(ct.GetSpacing())
    pet_position_max = pet_origin + np.array(pet.GetSize()) * np.array(pet.GetSpacing())
    
    #############################################################
    bb_min = np.maximum(ct_origin, pet_origin)
    bb_max = np.minimum(ct_position_max, pet_position_max)
    bb = np.concatenate([bb_min,
                         bb_max,], axis=0,
                        )

    size = np.round((bb[3:] - bb[:3]) / resampledPixelSpacing).astype(int)

    if np.any(size <= 0):
        bb_min = np.minimum(ct_origin, pet_origin)
        bb_max = np.maximum(ct_position_max, pet_position_max)
        bb = np.concatenate([bb_min,
                             bb_max,], axis=0,
                            )
    
        size = np.round((bb[3:] - bb[:3]) / resampledPixelSpacing).astype(int)
    
    
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])  
    
    resampler.SetInterpolator(sitk.sitkBSpline)
    pet_res = resampler.Execute(pet)
    ct_res = resampler.Execute(ct)
    
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_res = resampler.Execute(mask)
    
    return pet_res, ct_res, mask_res





def get_pet_based_bb(pet, image_size, th):


    np_pt = np.transpose(sitk.GetArrayFromImage(pet), (2, 1, 0))
    px_spacing_pt = pet.GetSpacing()
    px_origin_pt = pet.GetOrigin()

    # output_shape_pt = tuple(e1 / e2
    #                         for e1, e2 in zip(output_shape, px_spacing_pt))

    output_shape_pt = image_size

    # Gaussian smooth
    np_pt_gauss = gaussian_filter(np_pt, sigma=3)

    # OR fixed threshold
    np_pt_thgauss = np.where(np_pt_gauss > th, 1, 0)
    # Find brain as biggest blob AND not in lowest third of the scan
    labeled_array, _ = label(np_pt_thgauss)

    try:
        np_pt_brain = labeled_array == np.argmax(np.bincount(labeled_array[:, :,
                                                 np_pt.shape[2] * 2 // 3:].flat)[1:]) + 1
    except:
        print('th too high?')
        # Quick fix just to pass for all cases
        th = 0.1
        np_pt_thgauss = np.where(np_pt_gauss > th, 1, 0)
        labeled_array, _ = label(np_pt_thgauss)
        check_pt = np.bincount(labeled_array[:, :,
                                      np_pt.shape[2] * 2 // 3:].flat)[1:]
        if check_pt.size == 0 or check_pt.sum() == 0:
            print('insufficient bounding box')
            np_pt_brain = np.ones_like(np_pt, dtype=bool)
        else:
            np_pt_brain = labeled_array == np.argmax(check_pt) + 1

    # Find lowest voxel of the brain and box containing the brain
    z = np.min(np.argwhere(np.sum(np_pt_brain, axis=(0, 1))))
    y1 = np.min(np.argwhere(np.sum(np_pt_brain, axis=(0, 2))))
    y2 = np.max(np.argwhere(np.sum(np_pt_brain, axis=(0, 2))))
    x1 = np.min(np.argwhere(np.sum(np_pt_brain, axis=(1, 2))))
    x2 = np.max(np.argwhere(np.sum(np_pt_brain, axis=(1, 2))))

    # Center bb based on this brain segmentation
    zshift = 30 // px_spacing_pt[2]
    if z - (output_shape_pt[2] - zshift) < 0:
        zbb = (0, output_shape_pt[2])
    elif z + zshift > np_pt.shape[2]:
        zbb = (np_pt.shape[2] - output_shape_pt[2], np_pt.shape[2])
    else:
        zbb = (z - (output_shape_pt[2] - zshift), z + zshift)

    yshift = 30 // px_spacing_pt[1]
    if int((y2 + y1) / 2 - yshift - int(output_shape_pt[1] / 2)) < 0:
        ybb = (0, output_shape_pt[1])
    elif int((y2 + y1) / 2 - yshift -
                int(output_shape_pt[1] / 2)) > np_pt.shape[1]:
        ybb = np_pt.shape[1] - output_shape_pt[1], np_pt.shape[1]
    else:
        ybb = ((y2 + y1) / 2 - yshift - output_shape_pt[1] / 2,
               (y2 + y1) / 2 - yshift + output_shape_pt[1] / 2)

    if int((x2 + x1) / 2 - int(output_shape_pt[0] / 2)) < 0:
        xbb = (0, output_shape_pt[0])
    elif int((x2 + x1) / 2 -
                int(output_shape_pt[0] / 2)) > np_pt.shape[0]:
        xbb = np_pt.shape[0] - output_shape_pt[0], np_pt.shape[0]
    else:
        xbb = ((x2 + x1) / 2 - output_shape_pt[0] / 2,
               (x2 + x1) / 2 + output_shape_pt[0] / 2)

    z_pt = np.asarray(zbb)
    y_pt = np.asarray(ybb)
    x_pt = np.asarray(xbb)

    # In the physical dimensions
    z_abs = z_pt * px_spacing_pt[2] + px_origin_pt[2]
    y_abs = y_pt * px_spacing_pt[1] + px_origin_pt[1]
    x_abs = x_pt * px_spacing_pt[0] + px_origin_pt[0]

    bb = np.asarray((x_abs, y_abs, z_abs)).flatten()

    return bb






def crop_image_pet_based_bb(image, bb, crop_size):

    origin_cropped_1 = np.array([bb[0], bb[2], bb[4]]) 

    origin_cropped_2 = np.array([bb[1], bb[3], bb[5]]) 

    image_array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    image_shape = image_array.shape

    index_1 = list(image.TransformPhysicalPointToIndex(origin_cropped_1))
    index_2 = list(image.TransformPhysicalPointToIndex(origin_cropped_2))

    for i in range(3):
        if index_1[i] < 0:
            index_1[i] = 0
        if index_2[i] >= image_shape[i]:
            index_2[i] = image_shape[i] - 1


    image_array = image_array[int(index_1[0]):int(index_2[0]), 
                              int(index_1[1]):int(index_2[1]), 
                              int(index_1[2]):int(index_2[2])]

    image_out = sitk.GetImageFromArray(np.transpose(image_array, (2, 1, 0)))

    image_out.SetOrigin(image.TransformIndexToPhysicalPoint(index_1))

    image_out.SetSpacing(image.GetSpacing())
    
    
    # Pad if needed.
    if np.all(image_out.GetSize() == crop_size):
        cropped_image = image_out
    else:
        resize_transform = tio.CropOrPad(target_shape = crop_size,
                                          padding_mode = 0)

        cropped_image = resize_transform(image_out)

    return cropped_image





def get_mask_based_bb(mask, crop_size):

    mask_array = sitk.GetArrayFromImage(mask)

    # Get bounding box of the mask.
    non_zero_coords = np.array(np.nonzero(mask_array))
    min_coords = np.min(non_zero_coords, axis=1)
    max_coords = np.max(non_zero_coords, axis=1)

    # Compute center of the tumor.
    center = ((min_coords + max_coords) / 2).astype(int)

    # Compute crop region.
    start = center - crop_size // 2
    end = start + crop_size

    # Ensure within bounds.
    start = np.maximum(start, 0)
    end = np.minimum(end, mask_array.shape)
    
    return start, end





def crop_image_mask_based_bb(image, start, end, crop_size):
    
    array = sitk.GetArrayFromImage(image)
    cropped = array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    cropped_image = sitk.GetImageFromArray(cropped)
    
    # Set spatial info
    original_origin = image.TransformIndexToPhysicalPoint([int(start[2]), int(start[1]), int(start[0])])
    cropped_image.SetOrigin(original_origin)
    cropped_image.SetSpacing(image.GetSpacing())
    cropped_image.SetDirection(image.GetDirection())
    
    # Pad if needed.
    resize_transform = tio.CropOrPad(target_shape = crop_size,
                                      padding_mode = 0)
    
    cropped_image = resize_transform(cropped_image)
    
    return cropped_image




def make_surv_array(time, event, time_interval):
    '''
    Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        time: Array of failure/censoring times.
        event: Array of censoring indicator. 1 if failed, 0 if censored.
        time_interval: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        surv_array: Dimensions with (number of samples, number of time intervals*2)
    '''
    
    breaks = np.array(time_interval)
    n_intervals = len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    
    surv_array = np.zeros((n_intervals * 2))
    
    if event == 1:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks[1:]) 
        if time < breaks[-1]:
            surv_array[n_intervals + np.where(time < breaks[1:])[0][0]] = 1
    else:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks_midpoint)
    
    return surv_array





def data_preprocessing(sample_path):
    
    """
    - PET to CT registeration,
    - Resampling,
    - 
    """
        
    #if image_spacing is None:
    #   image_spacing = np.array([2, 2, 2])
    #if image_size is None:
    #   image_size = np.array([128, 128, 128])
        
        
    pet_orig = sitk.ReadImage(os.path.join(sample_path, 'PT_Pre_Ver2.nii.gz'))
    ct_orig = sitk.ReadImage(os.path.join(sample_path, 'CT_Pre_Ver2.nii.gz'))
    roi_orig = sitk.ReadImage(os.path.join(sample_path, 'ROI_Binary_Pre_Ver2.nii.gz'))
    df_orig = pd.read_excel(os.path.join(sample_path, 'Clinical_Total.xlsx'))

        
#     pet_orig = sitk.ReadImage(os.path.join(sample_path, 'PT_Reg.nii.gz')) # PT.nii.gz
#     ct_orig = sitk.ReadImage(os.path.join(sample_path, 'CT.nii.gz'))
#     roi_orig = sitk.ReadImage(os.path.join(sample_path, 'ROI_Binary.nii.gz'))
#     df_orig = pd.read_excel(os.path.join(sample_path, 'Clinical_Total.xlsx'))
    
    
##     # Apply registeration.
##     pet_reg_orig = registeration_func(moving_image = pet_orig, 
##                                       fixed_image = ct_orig, 
##                                       initializer_state=None)

    
#     # Apply Resampling.
#     pet_resample, ct_resample, roi_resample = hecktor_reampling(pet = pet_reg_orig, 
#                                                                 ct = ct_orig, 
#                                                                 mask = roi_orig, 
#                                                                 resampledPixelSpacing = image_spacing)
    
    
#     # Apply Resizing based on the bb corresponding to the PET intensity.
#     if resize_method == 'Crop':

#         bb = get_pet_based_bb(pet = pet_resample, 
#                               image_size = image_size, 
#                               th = 3)

#         pet_resize = crop_image_pet_based_bb(image = pet_resample, 
#                                              bb = bb,
#                                              crop_size = image_size)
        
#         ct_resize = crop_image_pet_based_bb(image = ct_resample, 
#                                              bb = bb,
#                                             crop_size = image_size)

#         roi_resize = crop_image_pet_based_bb(image = roi_resample, 
#                                              bb = bb,
#                                              crop_size = image_size)
        

#     elif resize_method == 'Resize':

#         resize_transform = tio.Resize(target_shape = image_size, 
#                                       label_interpolation = 'nearest',
#                                       image_interpolation = 'linear',)

#         pet_resize = resize_transform(pet_resample)
#         ct_resize = resize_transform(ct_resample)
#         roi_resize = resize_transform(roi_resample)
        

        
#     # Apply Normalizing.
#     pet_norm_transfrom = tio.RescaleIntensity(out_min_max = (0, 1), 
#                                               in_min_max = None)
#     pet_normalize = pet_norm_transfrom(pet_resize)


#     # Considering RescaleIntensity betwenn 0-1 for CT as well,
#     # So if we want to resize by padding, then default value of 0 works for both PET and CT.
#     ct_norm_transfrom = tio.Compose([tio.Clamp(out_min=-1000, out_max=1000),
#                                      tio.RescaleIntensity(out_min_max = (0, 1), 
#                                                           in_min_max = None),
#                                      # tio.ZNormalization(masking_method = None)
#                                     ])
#     ct_normalize = ct_norm_transfrom(ct_resize)


    
    
#     final_dic = {'PT': sitk.GetArrayFromImage(pet_normalize),
#                  'CT': sitk.GetArrayFromImage(ct_normalize),
#                  'Seg': sitk.GetArrayFromImage(roi_resize),
#                  'Clinic': df_orig}
    
    final_dic = {'PT': sitk.GetArrayFromImage(pet_orig),
                 'CT': sitk.GetArrayFromImage(ct_orig),
                 'Seg': sitk.GetArrayFromImage(roi_orig),
                 'Clinic': df_orig}
    
    return final_dic





def data_gen_ver1(vol_names, batch_size, balance_class=True, augmentaion=False, Sort=True):
    
    # Generate class index list.
    pos_indices = []
    neg_indices = []
    for idx in range(len(vol_names)):
        df = pd.read_excel(os.path.join(vol_names[idx], 'Clinical_Total.xlsx'))
        idx_Event = df['Relapse'].to_list()[0]
        if idx_Event == 1:
            pos_indices.append(idx)
        if idx_Event == 0:
            neg_indices.append(idx)

    while True:
        
        if balance_class == True:
            # manually balance class.
            half_batch = batch_size // 2
            if (len(pos_indices) < half_batch) or (len(neg_indices) < half_batch):
                raise ValueError('Not enough samples in one of the classes to create a balanced batch without replacement.')
            
            idxes_pos = np.random.choice(pos_indices, size=int(half_batch), replace=False)
            idxes_neg = np.random.choice(neg_indices, size=int(half_batch), replace=False)
            idxes = list(idxes_pos) + list(idxes_neg)
            np.random.shuffle(idxes)
        else:
            if len(vol_names) < batch_size:
                raise ValueError('Not enough total samples to generate batch without replacement.')
              
            idxes = np.random.choice(len(vol_names), size=batch_size, replace=False)

       
        # Preprocess data, and get them as dict.
        npz_data = [data_preprocessing(sample_path = vol_names[idx]) for idx in idxes]



        # Get PET, CT, and Seg.
        def stack_func(name):
            data = [d[name][np.newaxis, ..., np.newaxis] for d in npz_data]
            return np.concatenate(data, 0) if batch_size > 1 else data[0]

        PT = stack_func('PT')
        CT = stack_func('CT')
        Seg = stack_func('Seg')



        # Get Clinical label (time, event).
        label_data_classif = []
        label_data_surv = []
        for d in npz_data:
            df_clinic = d['Clinic']
            
            # Get lable for classification.
            hpv = df_clinic['HPV Status'].to_list()[0]
            x_hpv = np.array([hpv])
            x_hpv = x_hpv[np.newaxis, ...]
            label_data_classif.append(x_hpv)
            
            Time = df_clinic['RFS'].to_list()[0]
            Event = df_clinic['Relapse'].to_list()[0]
            X = np.array([Time, Event])
            X = X[np.newaxis, ...]
            label_data_surv.append(X)
            
        Label_classif = np.concatenate(label_data_classif, 0) if batch_size > 1 else label_data_classif[0]
        Label_surv = np.concatenate(label_data_surv, 0) if batch_size > 1 else label_data_surv[0]
        

        # Clinical features
        clinic_data = []
        for d in npz_data:
            df_clinic = d['Clinic']
            non_clinic_feature_name = ['PatientID', 'HPV Status', 'RFS', 'Relapse']
            X = df_clinic[[_ for _ in df_clinic.columns if _ not in non_clinic_feature_name]].to_numpy()
            clinic_data.append(X)
            # clinic_data.append(X[np.newaxis, ...])
        Clinic = np.concatenate(clinic_data, 0) if batch_size > 1 else clinic_data[0]


        #data augmentation
        if augmentaion == True:
            PT, CT, Seg = Data_augmentation(PT, CT, Seg)

        # Sort samples by survival time
        if Sort == True:
            PT = Sort_by_time(PT, Label_surv[:,0])
            CT = Sort_by_time(CT, Label_surv[:,0])
            Clinic = Sort_by_time(Clinic, Label_surv[:,0])
            
            Seg = Sort_by_time(Seg, Label_surv[:,0])
            #Event = Sort_by_time(Label_surv[:,1:], Label_surv[:,0])
            Surv = Sort_by_time(Label_surv, Label_surv[:,0])
            Label_classif = Sort_by_time(Label_classif, Label_surv[:,0])
            
        else:
            #Event = Label_surv[:,1:]
            Surv = Label_surv


        yield (PT, CT, Clinic), (Seg, Label_classif, Surv)   
        
        
        
        
def Data_augmentation(PT, CT, Seg):
    
    # define augmentation sequence
    aug_seq = iaa.Sequential([
        # horizontal flips
        iaa.Fliplr(0.5), 
        # translate/move them and rotate them.
        iaa.Affine(translate_px={"x": [-10, 10], "y": [0, 0]},rotate=(-5, 5))
        ],random_order=True) # apply augmenters in random order
    
    aug_seq_no_flip = iaa.Sequential([
        # translate/move them and rotate them.
        iaa.Affine(translate_px={"x": [-10, 10], "y": [0, 0]},rotate=(-5, 5))
        ],random_order=False)
    
    # pre-process data shape
    PT = PT[..., 0]
    CT = CT[..., 0]
    Seg = Seg[..., 0]
    
    # flip/translate in x axls, rotate along z axls
    images = np.concatenate((PT,CT,Seg), -1)
    
    images_aug = aug_seq(images=images)  # random flipping only occurs in the sagittal axis
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/3)]    
    CT = images_aug[..., int(images_aug.shape[3]/3):int(images_aug.shape[3]/3*2)]
    Seg = images_aug[..., int(images_aug.shape[3]/3*2):int(images_aug.shape[3])]
    
    # translate in z axls, rotate along y axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg = np.transpose(Seg,(0,3,1,2))
    images = np.concatenate((PT,CT,Seg), -1)
    
    images_aug = aug_seq_no_flip(images=images)
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/3)]    
    CT = images_aug[..., int(images_aug.shape[3]/3):int(images_aug.shape[3]/3*2)]
    Seg = images_aug[..., int(images_aug.shape[3]/3*2):int(images_aug.shape[3])]
    
    # translate in y axls, rotate along x axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg = np.transpose(Seg,(0,3,1,2))
    images = np.concatenate((PT,CT,Seg), -1)
    
    images_aug = aug_seq_no_flip(images=images) 
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/3)]    
    CT = images_aug[..., int(images_aug.shape[3]/3):int(images_aug.shape[3]/3*2)]
    Seg = images_aug[..., int(images_aug.shape[3]/3*2):int(images_aug.shape[3])]
    
    # recover axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg = np.transpose(Seg,(0,3,1,2))
    
    # reset Seg mask to 1/0
    for i in range(Seg.shape[0]):
        _, Seg[i] = cv2.threshold(Seg[i],0.2,1,cv2.THRESH_BINARY)
    
    # post-process data shape
    PT_aug = PT[..., np.newaxis]
    CT_aug = CT[..., np.newaxis]
    Seg_aug = Seg[..., np.newaxis]
    
    return PT_aug, CT_aug, Seg_aug





def Sort_by_time(data, time):
    '''
    Sort samples by survival time
    Designed for Cox loss function.
    '''
    sorted_arg = np.argsort(time)
    sorted_data = np.zeros(data.shape)
    
    for i in range(len(time)):
        sorted_data[i] = data[sorted_arg[i]]
        
    return sorted_data





def sort_data_gen(gen, augmentaion=False, Sort=True):
    while True:
        X = next(gen)
        PT = X[0]
        CT = X[1]
        Seg = X[2]
        Label = X[3]
        Clinic = X[4]
        
        #data augmentation
        if augmentaion == True:
            PT, CT, Seg = Data_augmentation(PT, CT, Seg)
        
        # Sort samples by survival time
        if Sort == True:
            PT = Sort_by_time(PT, Label[:,0])
            CT = Sort_by_time(CT, Label[:,0])
            Seg = Sort_by_time(Seg, Label[:,0])
            Event = Sort_by_time(Label[:,1:], Label[:,0])
            Clinic = Sort_by_time(Clinic, Label[:,0])
        else:
            Event = Label[:,1:]
        
        yield (PT, CT, Clinic), (Seg, Event)
        
        
        
        
def load_one_sample(vol_name):
    
    # Preprocess data, and get them as dict.
    sample_dic = data_preprocessing(sample_path = vol_name)

    # Get PET.
    PT = sample_dic['PT'][np.newaxis, ..., np.newaxis]
    
    # Get CT.
    CT = sample_dic['CT'][np.newaxis, ..., np.newaxis]
    
    # Get Seg.
    Seg = sample_dic['Seg'][np.newaxis, ..., np.newaxis]

    # Get Clinical label for classification and survival.
    df_clinic = sample_dic['Clinic']
    
    # Get lable for classification.
    hpv = df_clinic['HPV Status'].to_list()[0]
    Label_classif = np.array([hpv])
    Label_classif = Label_classif[np.newaxis, ...]
    
    
    # Get label for survival analysis.
    Time = df_clinic['RFS'].to_list()[0]
    Event = df_clinic['Relapse'].to_list()[0]
    Label_surv = np.array([Time, Event])
    Label_surv = Label_surv[np.newaxis, ...]

    # Clinical features.
    non_clinic_feature_name = ['PatientID', 'HPV Status', 'RFS', 'Relapse']
    Clinic = df_clinic[[_ for _ in df_clinic.columns if _ not in non_clinic_feature_name]].to_numpy()
    
    return PT, CT, Seg, Label_classif, Label_surv, Clinic
