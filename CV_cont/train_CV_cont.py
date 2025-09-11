# python imports
import os
import glob
import sys
import pdb
import random
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')

# third-party imports
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler


import csv
import cv2
import time
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index


# project imports
import datagenerators_hecktor
import networks_hecktor
import losses_hecktor
import metrics_hecktor



def get_y(path_list):
    
    y = []
    for path_sample in path_list:
        df_clinic = pd.read_excel(os.path.join(path_sample, 'Clinical_Total.xlsx'))
        event = df_clinic['Relapse'].to_list()[0]
        y.append(event)
        
    return np.array(y)




def get_survival_time(y_pred):

    breaks = np.array([0, 107, 195, 250, 325, 405, 526, 711, 923, 1395, 5888])
    
    intervals = breaks[1:] - breaks[:-1]
    n_intervals = len(intervals)
    
    Survival_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(y_pred[0:i+1])
        Survival_time = Survival_time + cumulative_prob * intervals[i]
    
    return Survival_time





def get_nan_zero_seg(path_list, image_spacing, image_size, resize_method):
    
    path_list_modif = list(path_list)
    for sample_path in path_list:
        
        print('\n', sample_path)
        
        sample_dic = datagenerators_hecktor.data_preprocessing(sample_path = sample_path)

        roi_all_zeros = np.all(sample_dic['Seg'] == 0)
        
        if roi_all_zeros:
            print('HERE.')
            # Remove the sample from the list.
            path_list_modif.remove(sample_path)
            
    return np.array(path_list_modif)
            




def save_excel_file(df_to_save, path_save_excel, sheet_name, index=False):
    # Write in excel.
    if os.path.exists(path_save_excel) is not True:
        with pd.ExcelWriter(path_save_excel) as writer:
            df_to_save.to_excel(writer, sheet_name=sheet_name, index=index)
    else:
        with pd.ExcelWriter(path_save_excel, mode='a', engine='openpyxl') as writer:
            df_to_save.to_excel(writer, sheet_name=sheet_name, index=index)
            
            
            
            
            
def lr_scheduler(epoch):

    if epoch < 50:
        lr = 1e-4
    elif epoch < 100:
        lr = 5e-5
    elif epoch < 200:
        lr = 1e-5
    else:
        lr = 1e-6
    print('lr: %f' % lr)
    return lr


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
        
        
        
        
def dice(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)
    
    
    
    
    
def evaluation(net,
               device,
               vol_names,
               image_spacing, 
               image_size,
               resize_method):
    
    dic_res = {'Patients': [],
               'Survival_true_time': [],
               'Survival_true_event': [],
               'Survival_pred_risk': []}
    
    for vol in vol_names:
        
        # load subject.
        sample_name = vol.split('\\')[-1]
        
        print('sample_name: ', sample_name)
        
        PT, CT, Seg, Label_classif, Label_surv, Clinic = datagenerators_hecktor.load_one_sample(vol_name = vol)

        with tf.device(device):
            pred = net.predict([PT, CT, Clinic])

            Seg_pred = pred[0][0,...,0]
            Classification_pred = pred[1][0,0]
            Survival_pred = pred[2][0,0]

        
        # Getting the results.
        dic_res['Patients'].append(sample_name)
        
        dic_res['Survival_true_time'].append(Label_surv[0][0])
        dic_res['Survival_true_event'].append(Label_surv[0][1])
        
        dic_res['Survival_pred_risk'].append(Survival_pred)

        
    # Survival.
    cindex_val = concordance_index(np.array(dic_res['Survival_true_time']), 
                                   -np.array(dic_res['Survival_pred_risk']), 
                                   np.array(dic_res['Survival_true_event']))
    
    
    return cindex_val, dic_res




class TestMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, csv_path, steps):
        """
        Args:
            test_data: tf.keras.utils.Sequence or generator yielding (x, y)
            csv_path: Path to save the CSV log file
            steps: Optional, number of steps to evaluate (for infinite generators)
        """
        super().__init__()
        self.test_data = test_data
        self.csv_path = csv_path
        self.steps = steps
        self.test_logs = []

        # Track whether we've written headers yet.
        self.headers_written = os.path.exists(csv_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Evaluate the model.
        test_results = self.model.evaluate(self.test_data, 
                                                     steps = self.steps, 
                                                     verbose = 0)
        
        # Convert to dict.
        dic_test_results = {out: test_results[i] for i, out in enumerate(self.model.metrics_names)}

        self.test_logs.append(dic_test_results)

        # Add epoch info.
        test_results_with_epoch = {'epoch': epoch}
        test_results_with_epoch.update(dic_test_results)

        # Write to CSV.
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=test_results_with_epoch.keys())

            if not self.headers_written:
                writer.writeheader()
                self.headers_written = True

            writer.writerow(test_results_with_epoch)
            

        # Print results.
        print('\ntest metrics:')
        for metric_name, metric_value in dic_test_results.items():
            print('test_%s: %.4f'%(metric_name, metric_value))   
        
        
        
        
class EpochTimerCSV(tf.keras.callbacks.Callback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

        # Create CSV and write header if file does not exist.
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Time (seconds)'])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        print('Total time for epoch %i: %.3f seconds.'%(epoch, elapsed))
        
        print('\n')

        # Append to CSV.
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(elapsed, 2)])




        
        
        
        
def train(data_dir,
          model_dir, 
          n_splits,
          random_state,
          resize_method,
          device,
          lr,
          nb_epochs,
          batch_size,
          batch_val_size,
          load_model_file,
          initial_epoch):
    
    
    # Image spacing.
    vol_spacing = np.array([2, 2, 2])
    # Image size.
    vol_size = np.array([128, 128, 128])
    # Clinical feature size.
    # Age, Gender, Tobacco Consumption, Alcohol Consumption, Performance Status, Treatment, M-stage.
    # With putting zero for missing data.
    Clinic_size = 7
    
    
    # Get volum paths.
    all_vol_names_orig = np.array([os.path.join(data_dir, _) for _ in os.listdir(data_dir)])
    

    # Keep only samples with nan-zero ROI after resizing.
    if resize_method == 'Crop':
        
        all_vol_names = all_vol_names_orig
        
    elif resize_method == 'Resize':
        
        all_vol_names = get_nan_zero_seg(path_list = all_vol_names_orig, 
                                         image_spacing = vol_spacing, 
                                         image_size = vol_size, 
                                         resize_method = resize_method)
    
    
    
    all_vol_y_event = get_y(path_list = all_vol_names)
    
    
    
    # CV with stratification ---> Train/Test.
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold_idx, (train_idx, validation_idx) in enumerate(skf.split(all_vol_names, all_vol_y_event)):
        
        print('==== Fold: %i ==='%(fold_idx+1))
        
        path_fold_save = os.path.join(model_dir, 'Fold%i'%(fold_idx+1))
        if os.path.exists(path_fold_save) is not True:
            os.makedirs(path_fold_save)
        
        # Get all training set.
        train_vol_names = all_vol_names[train_idx]
        random.seed(random_state)
        random.shuffle(train_vol_names)  
        

        # Get test set.
        validation_vol_names = all_vol_names[validation_idx]
        random.seed(random_state)
        random.shuffle(validation_vol_names)

        

        
        ## device handling
        if 'gpu' in device:
            if '0' in device:
                device = '/gpu:0'
            elif '1' in device:
                device = '/gpu:1'
    
            # TensorFlow 2.x equivalent for GPU config
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
        else:
            device = '/cpu:0'


        # prepare the model
        if 'gpu' in device:
            #strategy =  tf.distribute.MirroredStrategy()
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    
        # Prepare the model inside strategy scope
        with strategy.scope():
            model = networks_hecktor.DeepMTS(vol_size, Clinic_size)
    
            # load initial weights if available
            if load_model_file != './' and load_model_file is not None:
                print('loading', load_model_file)
                model.load_weights(load_model_file, by_name=True)
            print("Model metric names:", model.metrics_names)  
    
            model.compile(optimizer = Adam(learning_rate=lr), 
                          metrics = {'Segmentation': metrics_hecktor.Dice_coeff, 
                                     'Classifier': metrics_hecktor.Balanced_accuracy,
                                     'Survival': metrics_hecktor.Cindex},
                          loss = [losses_hecktor.Seg_loss, 
                                  losses_hecktor.Binary_focal_crossentropy,
                                  losses_hecktor.Cox_loss],
                          loss_weights = [1.0, 1.0, 1.0])   


        # data generator
        output_signature = (
        (
            tf.TensorSpec(shape=(None, *vol_size, 1), dtype=tf.float32),  # PET
            tf.TensorSpec(shape=(None, *vol_size, 1), dtype=tf.float32),  # CT
            tf.TensorSpec(shape=(None, Clinic_size), dtype=tf.float32)    # Clinic
        ),
        (
            tf.TensorSpec(shape=(None, *vol_size, 1), dtype=tf.float32),  # Segmentation mask
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),    # Classification label
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)    # Survival
        )
                            )
        
        def wrap_generator(vol_names, batch_size, balance_class, augmentaion, Sort):
            return tf.data.Dataset.from_generator(
                lambda: datagenerators_hecktor.data_gen_ver1(
                    vol_names=vol_names,
                    batch_size=batch_size,
                    balance_class=balance_class,
                    augmentaion=augmentaion,
                    Sort = Sort
                ),
                output_signature=output_signature
            )
    
        data_gen_train = wrap_generator(
        vol_names=train_vol_names,
        batch_size=batch_size,
        balance_class=True,
        augmentaion=False,
        Sort = True)
        # data_gen_train = data_gen_train.shuffle(buffer_size=100).prefetch(tf.data.AUTOTUNE)
        
        data_gen_valid = wrap_generator(
        vol_names=validation_vol_names,
        batch_size=batch_val_size,
        balance_class=True,   # Check this?
        augmentaion=False,
        Sort = True)



        path_model_save = os.path.join(path_fold_save, 'Model')
        if os.path.exists(path_model_save) is not True:
            os.makedirs(path_model_save)

        # Callback settings.
        save_file_name = os.path.join(path_model_save, 
         '{epoch:02d}-{val_Segmentation_dice_coeff:.3f}-{val_Classifier_balanced_accuracy:.3f}-{val_Survival_cindex:.3f}.weights.h5')
    
        # check here if the model is needed as input
        save_callback = tf.keras.callbacks.ModelCheckpoint(
                        save_file_name,
                        monitor='val_Survival_cindex',
                        save_best_only=False,
                        save_weights_only=True,
                        mode='max',
                        save_freq='epoch')

   

        path_log_save = os.path.join(path_fold_save, 'Log')
        if os.path.exists(path_log_save) is not True:
            os.makedirs(path_log_save)

        train_val_log_name = os.path.join(path_log_save, 'Log_Train_Val.csv')
        csv_logger = CSVLogger(train_val_log_name, append=True)

        early_stopping = EarlyStopping(monitor='val_Survival_Cindex', patience=100, mode='max')
        scheduler = LearningRateScheduler(lr_scheduler)


        time_log_name = os.path.join(path_log_save, 'Log_Time.csv')
        time_callback = EpochTimerCSV(csv_path = time_log_name)
    


        # compile settings and fit
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            callbacks = [save_callback, csv_logger, scheduler, early_stopping] # early_stopping
        else:
            callbacks = [save_callback, csv_logger, scheduler, time_callback, early_stopping] # early_stopping

        print("Steps per epoch:", len(train_vol_names) // batch_size)
        print("Validation steps:", len(validation_vol_names) // batch_val_size)


        # fit model
        model.fit(data_gen_train,
                            steps_per_epoch = len(train_vol_names) // batch_size,
                            validation_data = data_gen_valid,
                            validation_steps = len(validation_vol_names) // batch_val_size,
                            initial_epoch = initial_epoch,
                            epochs = nb_epochs,
                            callbacks = callbacks, # early_stopping
                            verbose = 1
                           )

        

#         path_res_save = os.path.join(path_fold_save, 'Result')
#         if os.path.exists(path_res_save) is not True:
#             os.makedirs(path_res_save)


#         # Evaluation over training set.
#         print('\nEvaluation over training set:')
#         cindex_val_train, dic_res_train = evaluation(net = model,
#                                                    device = device,
#                                                    vol_names = train_vol_names, 
#                                                    image_spacing = vol_spacing, 
#                                                    image_size = vol_size,
#                                                    resize_method = resize_method)

#         print('\nCindex Train: %.3f' %cindex_val_train)

#         df_res_train = pd.DataFrame(dic_res_train)
#         df_res_train.to_excel(os.path.join(path_res_save, 'Train_result.xlsx'),
#                               index=False)



#         # Evaluation over validation set.
#         print('\nEvaluation over validation set:')
#         cindex_val_validation, dic_res_validation = evaluation(net = model,
#                                                                device = device,
#                                                                vol_names = validation_vol_names,
#                                                                image_spacing = vol_spacing, 
#                                                                image_size = vol_size,
#                                                                resize_method = resize_method)

#         print('\nCindex Validation: %.3f' %cindex_val_validation)

#         df_res_validation = pd.DataFrame(dic_res_validation)
#         df_res_validation.to_excel(os.path.join(path_res_save, 'Validation_result.xlsx'),
#                               index=False)

     
        
        
        
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--data_dir', 
                        type = str,
                        dest = 'data_dir', 
                        default = r'Hecktor_2025/Training data/Data_02072025',
                        help = 'Data folder')

    parser.add_argument('--model_dir', 
                        type = str,
                        dest = 'model_dir', 
                        default = r'Hecktor_2025/Model_Res/Continuous_Modif_trainval_RFS_test',
                        help = 'Model folder')
    
    parser.add_argument('--n_splits', 
                        type = int,
                        dest = 'n_splits', 
                        default = 3,
                        help = 'Number of splits')

    parser.add_argument('--random_state', 
                        type = int,
                        dest = 'random_state', 
                        default = 234,
                        help = 'Seed')

    parser.add_argument('--resize_method', 
                        type = str,
                        dest = 'resize_method', 
                        default = 'Crop',
                        help = 'Resizing method')

    parser.add_argument('--device', 
                        type = str,
                        dest = 'device', 
                        default = 'gpu1',
                        help = 'Device')

    parser.add_argument('--lr', 
                        type = float,
                        dest = 'lr', 
                        default = 1e-4, 
                        help = 'Learning rate')

    parser.add_argument('--epochs', 
                        type = int,
                        dest = 'nb_epochs', 
                        default = 500, #500,
                        help = 'Number of epoch')

    parser.add_argument('--batch_size', 
                        type = int,
                        dest = 'batch_size', 
                        default = 4,
                        help = 'Train batch size')

    parser.add_argument('--batch_val_size', 
                        type = int,
                        dest = 'batch_val_size', 
                        default = 16,
                        help = 'Validation batch size')

    parser.add_argument('--load_model_file', 
                        type = str,
                        dest = 'load_model_file', 
                        default = './',
                        help = 'Optional h5 model file to initialize with')

    parser.add_argument('--initial_epoch', 
                        type = int,
                        dest = 'initial_epoch', 
                        default = 0,
                        help = 'initial epoch')


    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)  # Ensure dir exists
    args_txt_path = os.path.join(args.model_dir, 'Args.txt')
    with open(args_txt_path, 'w') as f:
        # json.dump(args.__dict__, f, indent=2)
        json.dump(
    {k: v for k, v in args.__dict__.items() if not isinstance(v, tf.data.Dataset)},
    f,
    indent=2)

    train(**vars(args))