# -*- coding: utf-8 -*-
"""
Preprocessing script
Created: 2020/03/13
Updated: 2020/03/14 : NOTES : added patient labels, hypnogram and subject loop
By: Leonore Bovy

Script to cut the input (EDF) into 30 second segments

"""

## Install MNE package from here: https://mne.tools/dev/install/mne_python.html

import mne
import numpy as np
from   numpy import loadtxt
import h5py
import time

## Read in patient labels
pat_labels = loadtxt("P:/3013065.04/ml_project/patient_labels.txt", delimiter="\t", skiprows = 1)

response = pat_labels[:,1]
#subjects = pat_labels[:,0]
subjects = [2]

for idx, c_subj in enumerate(subjects):
    print (f'index: {idx} and c{c_subj}')
    ## Read in data
    file     = "P:/3013065.04/ml_project/test_data/LP_" + str(int(c_subj)) + "_1.edf"
    tic      = time.time()
    data     = mne.io.read_raw_edf(file)
    raw_data = data.get_data()
    print('Time to read EDF: {}'.format(time.time()-tic))
    
    ## Get channel info
    info     = data.info
    channels = data.ch_names
    
    ## Remove certain unwanted channels (mastoids); What about EMG, ECG?
    # ...
    
    ## Sampling rate is 200hz; thus 1 epoch(30s) is 6000 samples
    len_epoch   = 6000
    start_epoch = 0
    n_channels  =  len(channels)
       
    ## Cut tail; use modulo to find full epochs
    raw_data = raw_data[:, 0:raw_data.shape[1] - raw_data.shape[1]%len_epoch]
    
    ## Reshape data [n_channel, len_epoch, n_epochs]
    data_epoched = np.reshape(raw_data, (n_channels, len_epoch, int(raw_data.shape[1]/len_epoch)), order='F' )
    data_label   = np.ones((data_epoched.shape[2],1))*response[idx]
    
    ## Read in hypnogram data
    hyp = loadtxt("P:/3013065.04/Depressed_Loreta/hypnograms/LP_" + str(int(c_subj)) + "_1.txt", delimiter="\t")
    
    ## Create sepereate data subfiles based on hypnogram (N1, N2, N3, NREM, REM)
    ...
    
    
'''
## Save data
fname = 'LP_' + str(int(c_subj))
with h5py.File(fname + '.h5', 'a') as wf:
    dset = wf.create_dataset('data', data_epoched.shape, data=data_epoched)
    dset = wf.create_dataset('label', data_label.shape, data=data_label)

## Read data
tic = time.time()    
with h5py.File('P:/3013065.04/ml_project/data/LP_2.h5', 'r') as rf:
    x = rf['data'][:]
print('Time to read H5: {}'.format(time.time()-tic))'''