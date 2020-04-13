# -*- coding: utf-8 -*-
"""
Preprocessing script
Created: 2020/03/18

Script to concatenate all the saved REM epochs of control and subject groups,
split train and test sets and prepare for CNN classification.
"""

import numpy as np
from   numpy import loadtxt
import h5py
import time
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

## Define butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
def create_train_test_splits(stage, path, additional_path, saving_name, saving_path,
                             ch1_name, ch2_name,
                             patient_control_labels="P:/3013080.02/ml_project/grouping.txt",
                             train_size=.9):
    ## Read in patient labels
    #pat_labels = loadtxt("P:/3013080.02/ml_project/patient_labels.txt", delimiter="\t", skiprows = 1)
    
    # Distinguishing patients from control group
    gp = loadtxt(patient_control_labels, delimiter="\t", skiprows = 1, dtype = 'str')
    subj_c = [] # Control
    subj_p = [] # Patients
    
    # Creating standard scaler object
    sc_Fp1 = StandardScaler()
    sc_Fp2 = StandardScaler()
    
    for indx, c in enumerate(gp):
        if c[1] == 'C':
            subj_c.append(int(c[0]))
        elif c[1] == 'CC':
            pass
        else:
            subj_p.append(int(c[0]))
    
    # Detect number of Controls and patients 
    n_c = len(subj_c)
    n_p = len(subj_p)
    
    # Train vs Test Proportion
    train_size = train_size
    
    # Amount of train/test sets per group
    n_train_c = round(train_size * n_c)
    n_train_p = round(train_size * n_p)
    n_test_c  = len(subj_c) - n_train_c
    n_test_p  = len(subj_p) - n_train_p
    
    # Random permutation to separate train and test sets
    subj_c = np.random.RandomState(seed=42).permutation(subj_c)
    subj_p = np.random.RandomState(seed=42).permutation(subj_p)
    
    # Initializing train / test splits
    x_train_ch1 = np.empty((0,6000))
    x_train_ch2 = np.empty((0,6000))
    x_test_ch1  = np.empty((0,6000))
    x_test_ch2  = np.empty((0,6000))
    ############################## TRAINING SET ##################################
    # Read CONTROL group data for TRAIN
    tic = time.time()    
    for i in subj_c[0:n_train_c]:
        fname = (path + stage + additional_path+'/LK_' + str(i) + '_1.h5')
        with h5py.File(fname, 'r') as rf:
            tmp = rf['.']['data'].value
        tmp_ch1 = np.transpose(tmp[0,:,:])
        tmp_ch2 = np.transpose(tmp[1,:,:])
        x_train_ch1 = np.append(x_train_ch1, tmp_ch1, axis = 0)
        x_train_ch2 = np.append(x_train_ch2, tmp_ch2, axis = 0)
    print('Control data for training was loaded in : {} secs'.format(time.time()-tic))
    
    # Create output labels for 
    y_train_ch1 = np.zeros((np.shape(x_train_ch1)[0],2))
    y_train_ch2 = np.zeros((np.shape(x_train_ch2)[0],2))
    
    # FIRST column is CNOTROL class
    y_train_ch1[:,0] = 1
    y_train_ch2[:,0] = 1
    
    # Read PATIENT group data for TRAIN
    tic = time.time() 
    for i in subj_p[0:n_train_p]:
        fname = (path + stage +additional_path + '/LP_' + str(i) + '_1.h5')
        with h5py.File(fname, 'r') as rf:
            tmp = rf['.']['data'].value
        tmp_ch1 = np.transpose(tmp[0,:,:])
        tmp_ch2 = np.transpose(tmp[1,:,:])
        x_train_ch1 = np.append(x_train_ch1, tmp_ch1, axis = 0)
        x_train_ch2 = np.append(x_train_ch2, tmp_ch2, axis = 0)
    print('Patients data for training was loaded in : {} secs'.format(time.time()-tic))
    
    # Add abels for patients (second column)
    n_old = np.shape(y_train_ch1)[0]
    n_new = np.shape(x_train_ch1)[0] 
    y_train_ch1 = np.append(y_train_ch1, np.zeros((n_new - n_old,2)), axis = 0)
    y_train_ch2 = np.append(y_train_ch2, np.zeros((n_new - n_old,2)), axis = 0)
    y_train_ch1[n_old:,1] = 1
    y_train_ch2[n_old:,1] = 1
    
    
    ################################ TEST SET ####################################
    
    # Read CONTROL group data for TEST
    tic = time.time()    
    for i in subj_c[n_train_c:]:
        fname = (path + stage + additional_path + '/LK_' + str(i) + '_1.h5')
        with h5py.File(fname, 'r') as rf:
            tmp = rf['.']['data'].value
        tmp_ch1 = np.transpose(tmp[0,:,:])
        tmp_ch2 = np.transpose(tmp[1,:,:])
        x_test_ch1 = np.append(x_test_ch1, tmp_ch1, axis = 0)
        x_test_ch2 = np.append(x_test_ch2, tmp_ch2, axis = 0)
        
    print('Control data for testig was loaded in : {} secs'.format(time.time()-tic))
    
    # Create output labels for 
    y_test_ch1 = np.zeros((np.shape(x_test_ch1)[0],2))
    y_test_ch2 = np.zeros((np.shape(x_test_ch2)[0],2))
    
    # FIRST column is CONTROL class
    y_test_ch1[:,0] = 1
    y_test_ch2[:,0] = 1
    
    # Read PATIENT group data for TEST
    tic = time.time() 
    for i in subj_p[n_train_p:]:
        fname = (path + stage + additional_path + '/LP_' + str(i) + '_1.h5')
        with h5py.File(fname, 'r') as rf:
            tmp = rf['.']['data'].value
        tmp_ch1 = np.transpose(tmp[0,:,:])
        tmp_ch2 = np.transpose(tmp[1,:,:])
        x_test_ch1 = np.append(x_test_ch1, tmp_ch1, axis = 0)
        x_test_ch2 = np.append(x_test_ch2, tmp_ch2, axis = 0)
    print('Patients data for test was loaded in : {} secs'.format(time.time()-tic))
    
    # Add abels for patients (second column)
    n_old = np.shape(y_test_ch1)[0]
    n_new = np.shape(x_test_ch1)[0] 
    y_test_ch1 = np.append(y_test_ch1, np.zeros((n_new - n_old,2)), axis = 0)
    y_test_ch2 = np.append(y_test_ch2, np.zeros((n_new - n_old,2)), axis = 0)
    y_test_ch1[n_old:,1] = 1
    y_test_ch2[n_old:,1] = 1
    
    print('Train and test splits have been successfully generated! \n')
    
    # SAVE train/test splits
    fname = (saving_path + saving_name)
    with h5py.File((fname + '.h5'), 'w') as wf:
        dset = wf.create_dataset('y_test_'+ch1_name, y_test_ch1.shape, data=y_test_ch1)
        dset = wf.create_dataset('y_test_'+ch2_name, y_test_ch2.shape, data=y_test_ch2)
        dset = wf.create_dataset('y_train_'+ch1_name, y_train_ch1.shape, data=y_train_ch1)
        dset = wf.create_dataset('y_train_'+ch2_name, y_train_ch2.shape, data=y_train_ch2)
        dset = wf.create_dataset('x_test_'+ch1_name, x_test_ch1.shape, data=x_test_ch1)
        dset = wf.create_dataset('x_test_'+ch2_name, x_test_ch2.shape, data=x_test_ch2)
        dset = wf.create_dataset('x_train_'+ch1_name, x_train_ch1.shape, data=x_train_ch1)
        dset = wf.create_dataset('x_train_'+ch2_name, x_train_ch2.shape, data=x_train_ch2)
    print('Time to save H5: {}'.format(time.time()-tic))


# Apply function
create_train_test_splits(stage = 'REM',
                         path='D:/1D_TimeSeries/raw_EEG/without artefact/O1-M2_O2-M1/',
                         additional_path='', 
                         saving_name='tr90_REM_O1-M2_O2-M1', 
                         saving_path = 'D:/1D_TimeSeries/raw_EEG/without artefact/train_test/',
                         ch1_name = 'O1-M2', ch2_name = 'O2-M1',
                         patient_control_labels="P:/3013080.02/ml_project/grouping.txt",
                         train_size=.9)