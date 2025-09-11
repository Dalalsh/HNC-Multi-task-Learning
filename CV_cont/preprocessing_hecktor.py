import os
import csv
import pdb
import time
import json
import copy
import shutil
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import torchio as tio
import pydicom as dicom
import SimpleITK as sitk

from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import label




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


def data_preprocessing(sample_path, image_spacing, image_size, resize_method='Crop'):
    
    """
    - PET to CT registeration,
    - Resampling,
    - 
    """
        
    if image_spacing is None:
        image_spacing = np.array([2, 2, 2])
    if image_size is None:
        image_size = np.array([128, 128, 128])

        
    pet_orig = sitk.ReadImage(os.path.join(sample_path, 'PT.nii.gz'))
    ct_orig = sitk.ReadImage(os.path.join(sample_path, 'CT.nii.gz'))
    roi_orig = sitk.ReadImage(os.path.join(sample_path, 'ROI_Binary.nii.gz'))
    df_orig = pd.read_excel(os.path.join(sample_path, 'Clinical_Total.xlsx'))
    
    
##     # Apply registeration.
##     pet_reg_orig = registeration_func(moving_image = pet_orig, 
##                                       fixed_image = ct_orig, 
##                                       initializer_state=None)

    
    # Apply Resampling.
    pet_resample, ct_resample, roi_resample = hecktor_reampling(pet = pet_orig, 
                                                                ct = ct_orig, 
                                                                mask = roi_orig, 
                                                                resampledPixelSpacing = image_spacing)
    
    
    # Apply Resizing based on the bb corresponding to the PET intensity.
    if resize_method == 'Crop':

        bb = get_pet_based_bb(pet = pet_resample, 
                              image_size = image_size, 
                              th = 3)

        pet_resize = crop_image_pet_based_bb(image = pet_resample, 
                                             bb = bb,
                                             crop_size = image_size)
        
        ct_resize = crop_image_pet_based_bb(image = ct_resample, 
                                             bb = bb,
                                            crop_size = image_size)

        roi_resize = crop_image_pet_based_bb(image = roi_resample, 
                                             bb = bb,
                                             crop_size = image_size)
        

    elif resize_method == 'Resize':

        resize_transform = tio.Resize(target_shape = image_size, 
                                      label_interpolation = 'nearest',
                                      image_interpolation = 'linear',)

        pet_resize = resize_transform(pet_resample)
        ct_resize = resize_transform(ct_resample)
        roi_resize = resize_transform(roi_resample)
        

        
    # Apply Normalizing.
    pet_norm_transfrom = tio.RescaleIntensity(out_min_max = (0, 1), 
                                              in_min_max = None)
    pet_normalize = pet_norm_transfrom(pet_resize)


    # Considering RescaleIntensity betwenn 0-1 for CT as well,
    # So if we want to resize by padding, then default value of 0 works for both PET and CT.
    ct_norm_transfrom = tio.Compose([tio.Clamp(out_min=-1000, out_max=1000),
                                     tio.RescaleIntensity(out_min_max = (0, 1), 
                                                          in_min_max = None),
                                     # tio.ZNormalization(masking_method = None)
                                    ])
    ct_normalize = ct_norm_transfrom(ct_resize)


    
    
    final_dic = {'PT': sitk.GetArrayFromImage(pet_normalize),
                 'CT': sitk.GetArrayFromImage(ct_normalize),
                 'Seg': sitk.GetArrayFromImage(roi_resize),
                 'Clinic': df_orig}
    
    return pet_normalize, ct_normalize, roi_resize


# Where the you retrived the clean data ---> 672 samples.
path_data = r'YOUR/PATH/TO/DATA'

image_spacing = np.array([2, 2, 2])
image_size = np.array([128, 128, 128])

for patient_name in os.listdir(path_data):
        
    print(patient_name)

    patient = os.path.join(path_data, patient_name)

    pet_pre, ct_pre, roi_pre = data_preprocessing(sample_path = patient, 
                                                  image_spacing = image_spacing, 
                                                  image_size = image_size, 
                                                  resize_method = 'Crop')


    sitk.WriteImage(pet_pre, os.path.join(patient, 'PT_Pre_Ver2.nii.gz'))
    sitk.WriteImage(ct_pre, os.path.join(patient, 'CT_Pre_Ver2.nii.gz'))
    sitk.WriteImage(roi_pre, os.path.join(patient, 'ROI_Binary_Pre_Ver2.nii.gz'))
