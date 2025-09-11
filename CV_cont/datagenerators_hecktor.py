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
        
    pet_orig = sitk.ReadImage(os.path.join(sample_path, 'PT_Pre_Ver2.nii.gz'))
    ct_orig = sitk.ReadImage(os.path.join(sample_path, 'CT_Pre_Ver2.nii.gz'))
    roi_orig = sitk.ReadImage(os.path.join(sample_path, 'ROI_Binary_Pre_Ver2.nii.gz'))
    df_orig = pd.read_excel(os.path.join(sample_path, 'Clinical_Total.xlsx'))

    
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
