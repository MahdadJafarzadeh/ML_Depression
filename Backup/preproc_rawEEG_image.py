# -*- coding: utf-8 -*-
"""
Preprocessing script
Created: 2020/03/13
Updated: 2020/03/14 : NOTES : added patient labels, hypnogram and subject loop
Updated: 2020/03/16 : NOTES : 1.saving raw EEG signals per subject,
                              per channel of interest, per epoch
                              2. Butterworth filter is added.
                             
By: Leonore Bovy

Script to cut the input (EDF) into 30 second segments

"""

## Install MNE package from here: https://mne.tools/dev/install/mne_python.html

import mne
import numpy as np
import matplotlib.pyplot as plt
from   numpy import loadtxt
import h5py
import time
import os
from IPython import get_ipython
from scipy.signal import butter, lfilter

## Figures in separate window (to enhance quality of image)
get_ipython().run_line_magic('matplotlib', 'qt')

## Define butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
 
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

    
    # Retrieve info
    DataInfo          = data.info
    AvailableChannels = DataInfo['ch_names']
    fs                = int(DataInfo['sfreq'])
    
    # Filtering
    tic      = time.time()
    #raw_data = butter_bandpass_filter(raw_data,fs=fs,lowcut=.3,highcut= 20)
    print('Filtering time: {}'.format(time.time()-tic))
    
    ## Choosing channels of interest (which ones to choose?)
    RequiredChannels = ['Fp1', 'Fp2'] # test
    # Find index of requiored channels
    Idx = []
    for indx, c in enumerate(AvailableChannels):
        if c in RequiredChannels:
            Idx.append(indx)
    
    ## Sampling rate is 200hz; thus 1 epoch(30s) is 6000 samples
    T = 30 #secs
    len_epoch   = fs * T
    start_epoch = 0
    n_channels  =  len(AvailableChannels)
       
    ## Cut tail; use modulo to find full epochs
    raw_data = raw_data[:, 0:raw_data.shape[1] - raw_data.shape[1]%len_epoch]
    
    ## Reshape data [n_channel, len_epoch, n_epochs]
    data_epoched = np.reshape(raw_data, (n_channels, len_epoch, int(raw_data.shape[1]/len_epoch)), order='F' )
    data_label   = np.ones((data_epoched.shape[2],1))*response[idx]
    
    ## Read in hypnogram data
    hyp = loadtxt("P:/3013065.04/Depressed_Loreta/hypnograms/LP_" + str(int(c_subj)) + "_1.txt", delimiter="\t")
    
    ### Create sepereate data subfiles based on hypnogram (N1, N2, N3, NREM, REM) 
    tic      = time.time()
    
    # Seprate channels of interest:
    data_epoched_selected = data_epoched[Idx]
    
    # N1
    N1_epochs_idx   = [i for i,j in enumerate(hyp[:,0]) if j == 1]
    N1_epochs       = data_epoched_selected[:,:, N1_epochs_idx]
    # N2
    N2_epochs_idx   = [i for i,j in enumerate(hyp[:,0]) if j == 2]
    N2_epochs       = data_epoched_selected[:,:, N2_epochs_idx]
    # N3
    N3_epochs_idx   = [i for i,j in enumerate(hyp[:,0]) if j == 3]
    N3_epochs       = data_epoched_selected[:,:, N3_epochs_idx]
    # REM
    REM_epochs_idx  = [i for i,j in enumerate(hyp[:,0]) if j == 5]
    REM_epochs      = data_epoched_selected[:,:, REM_epochs_idx]
    # NREM
    NREM_epochs_idx = N1_epochs_idx + N2_epochs_idx + N3_epochs_idx
    NREM_epochs     = data_epoched_selected[:,:, NREM_epochs_idx]
    # Task finished: generate message
    print('Time to split sleep stages per epoch: {}'.format(time.time()-tic))
    
    ### Check existence of required directories for saving plots
    tic      = time.time()
    folders = ['N1','N2','N3','REM','NREM']
    for j in np.arange(len(folders)):
        for jj in np.arange(len(RequiredChannels)):
            if not os.path.exists('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/'+
                                  folders[j] + '/' + RequiredChannels[jj]):
                os.makedirs('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/' +
                            folders[j] + '/' + RequiredChannels[jj])
                
    # Task finished: generate message            
    print('Required folders are created in {} secodns'.format(time.time()-tic))
                  
    ################# Generating plots per epoch and save them ################
    ''' SAVING FORMAT:
    FOLDER:Plots/raw_EEG/SubjectNo_(1:pre-medication or 2:post-medication)/SleepStage/Channel/
    FILE NAME: SubjectNo_(1:pre-medication or 2:post-medication)/SleepStage/EpochNo''' 
    
    for ch in np.arange(len(Idx)):
        
        tic      = time.time()
    # Current channel of choice
        curr_ch = AvailableChannels[Idx[ch]]
    
    ### Plot N1 epochs per channel and save them ###
        for jj in np.arange(0,np.shape(N1_epochs)[2]):
            plt.plot(N1_epochs[ch,:,jj])
            plt.xlim(left = 0, right = len_epoch)
            plt.axis('off')
            # This "ylim" boundary has been set for EEG, be careful about ECG, EMG!
            plt.ylim(bottom = -10**-4, top = 10**-4 )
            # save into N1 folder
            plt.savefig('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/N1/'+
                        curr_ch +'/S' + str(int(c_subj)) + '_1_N1_' + str(int(jj)) ,dpi = 95)
            # Save into NREM folder
            plt.savefig('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/NREM/'+
                        curr_ch +'/S' + str(int(c_subj)) + '_1_N1_' + str(int(jj)),dpi = 95)        
            plt.close()
        
        ### Plot N2 epochs per channel and save them ###
        for jj in np.arange(0,np.shape(N2_epochs)[2]):
            plt.plot(N2_epochs[ch,:,jj])
            plt.xlim(left = 0, right = len_epoch)
            plt.axis('off')
            # This "ylim" boundary has been set for EEG, be careful about ECG, EMG!
            plt.ylim(bottom = -10**-4, top = 10**-4 )
            # Save into N2 folder
            plt.savefig('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/N2/'+
                        curr_ch +'/S' + str(int(c_subj)) + '_1_N2_' + str(int(jj)),dpi = 95)
            # Save into NREM folder
            plt.savefig('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/NREM/'+
                        curr_ch +'/S' + str(int(c_subj)) + '_1_N2_' + str(int(jj)),dpi = 95)
            plt.close()
        
        ### Plot N3 epochs per channel and save them ###
        for jj in np.arange(0,np.shape(N3_epochs)[2]):
            plt.plot(N3_epochs[ch,:,jj])
            plt.xlim(left = 0, right = len_epoch)
            plt.axis('off')
            # This "ylim" boundary has been set for EEG, be careful about ECG, EMG!
            plt.ylim(bottom = -10**-4, top = 10**-4 )
            # Save into N3 folder
            plt.savefig('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/N3/'+
                        curr_ch +'/S' + str(int(c_subj)) + '_1_N3_'+ str(int(jj)) ,dpi = 95)
            # Save into NREM folder
            plt.savefig('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/NREM/'+
                        curr_ch +'/S' + str(int(c_subj)) + '_1_N3_'+ str(int(jj)) ,dpi = 95)        
            plt.close()
        
        ### Plot REM epochs per channel and save them ###
        for jj in np.arange(0,np.shape(REM_epochs)[2]):
            plt.plot(REM_epochs[ch,:,jj])
            plt.xlim(left = 0, right = len_epoch)
            plt.axis('off')
            # This "ylim" boundary has been set for EEG, be careful about ECG, EMG!
            plt.ylim(bottom = -10**-4, top = 10**-4 )
            # Save into REM folder
            plt.savefig('Plots/raw_EEG/S'+ str(int(c_subj)) + '_1/REM/'+
                        curr_ch +'/S' + str(int(c_subj)) + '_1_REM_'+ str(int(jj)) ,dpi = 95)
            plt.close()
            
    print('Finished! \nTime to create all figures: {}'.format(time.time()-tic))


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