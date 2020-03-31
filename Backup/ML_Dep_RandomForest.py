# -*- coding: utf-8 -*-
"""
Created on 26/03/2020 
@author: Mahdad

----
This script is to classify REM sleep epochs derived from depressed and normal 
subjects to classify the subject either as normal or depressed, using hand-crafted
fetures and supervised machine learning methods such as Random forest.

"""
#%% Loading data section
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
#from entropy.entropy import spectral_entropy
from sklearn.model_selection import GridSearchCV
from scipy.fftpack import fft
import h5py
import time


# Load data
tic = time.time() 
fname = ("P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/tr90_fp1-M2_fp2-M1.h5")

# choose channel to extract features from
ch = 'fp2-M1'
fs = 200 #Hz
T = 30 #sec
# Split train and test 
with h5py.File(fname, 'r') as rf:
    xtest  = rf['.']['x_test_' + ch].value
    xtrain = rf['.']['x_train_' + ch].value
    ytest  = rf['.']['y_test_' + ch].value
    ytrain = rf['.']['y_train_' + ch].value
print('train and test data loaded in : {} secs'.format(time.time()-tic))

# Flatten data for filter and normalization
X_train = np.reshape(xtrain, (np.shape(xtrain)[0] * np.shape(xtrain)[1] ,1))
X_test  = np.reshape(xtest, (np.shape(xtest)[0] * np.shape(xtest)[1] ,1))

#%% Filtering section
## Defining preprocessing function ##
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b,a)
    y = lfilter(b, a, data)
    return y

# Apply filter
X_train = butter_bandpass_filter(data=X_train, lowcut=.3, highcut=20, fs=fs, order=2)
X_test  = butter_bandpass_filter(data=X_test , lowcut=.3, highcut=20, fs=fs, order=2)

#%% Normalization section - DEACTIVATED
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test  = sc.transform(X_test)

#%% Reshaping data per epoch
X_train = np.reshape(X_train, (int(len(X_train) / (fs*T)), fs*T))
X_test  = np.reshape(X_test,  (int(len(X_test) / (fs*T)), fs*T))
X       = np.concatenate((X_train, X_test))
Y       = np.concatenate((ytrain, ytest))

#%% Feature Extraction section

# Defining EEG bands:
eeg_bands = {'Delta'     : (0.5, 4),
             'Theta'     : (4  , 8),
             'Alpha'     : (8  , 12),
             'Beta'      : (12 , 20),
             'Sigma'     : (12 , 16),
             'Sigma_slow': (12 , 14),
             'Sigma_fast': (14 , 16)}

# Initializing variables of interest
eeg_band_fft      = dict()
freq_ix           = dict()
Features = np.empty((0, 21))
# Settings of peridogram    
Window = 'hann'
# zero-padding added with respect to (Nfft=2^(nextpow2(len(window))))
Nfft = 2 ** 15 
# Defining freq. resoultion
fm, _ = periodogram(x = X[0,:], fs = fs, nfft = Nfft , window = Window)  

# Finding the index of different freq bands with respect to "fm" #
for band in eeg_bands:
    freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
                       (fm <= eeg_bands[band][1]))[0]    


# Defining for loop to extract features per epoch
for i in np.arange(len(X)):
    
    data = X[i,:]
    
    # Compute the "total" power inside the investigational window
    _ , pxx = periodogram(x = data, fs = fs, nfft = Nfft , window = Window) 
    
    # Initialization for wavelet 
    cA_values = []
    cD_values = []
    cA_mean   = []
    cA_std    = []
    cA_Energy = []
    cD_mean   = []
    cD_std    = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []

    '''Power in differnt freq ranges ''' 
    # Total pow is defined form 0.5 - 20 Hz
    pow_total      = np.sum(pxx[np.arange(freq_ix['Delta'][0], freq_ix['Beta'][-1]+1)])
    Pow_Delta      = np.sum(pxx[freq_ix['Delta']]) / pow_total
    Pow_Theta      = np.sum(pxx[freq_ix['Theta']]) / pow_total
    Pow_Alpha      = np.sum(pxx[freq_ix['Alpha']]) / pow_total
    Pow_Beta       = np.sum(pxx[freq_ix['Beta']])  / pow_total
    Pow_Sigma      = np.sum(pxx[freq_ix['Sigma']]) / pow_total
    Pow_Sigma_slow = np.sum(pxx[freq_ix['Sigma_slow']]) / pow_total
    Pow_Sigma_fast = np.sum(pxx[freq_ix['Sigma_fast']]) / pow_total
    
    '''bApply Welch to see the dominant freq band''' 
    #Pwelch = welch(x = data, fs = fs, window = 'hann', nperseg= 512, nfft = Nfft)
    
    ''' Spectral Entropy '''
    #Entropy_Welch = spectral_entropy(x = data, sf=fs, method='welch', nperseg = 512)
    #Entropy_fft   = spectral_entropy(x = data, sf=fs, method='fft')
    
    ''' Wavelet Decomposition ''' 
    cA,cD=pywt.dwt(data,'coif1')
    cA_values.append(cA)
    cD_values.append(cD)
    cA_mean.append(np.mean(cA_values))
    cA_std.append(np.std(cA_values))
    cA_Energy.append(np.sum(np.square(cA_values)))
    cD_mean.append(np.mean(cD_values))
    cD_std.append(np.std(cD_values))
    cD_Energy.append(np.sum(np.square(cD_values)))
    Entropy_D.append(np.sum(np.square(cD_values) * np.log(np.square(cD_values))))
    Entropy_A.append(np.sum(np.square(cA_values) * np.log(np.square(cA_values))))
    
    ''' Statisctical features'''
    Kurt     = kurtosis(data, fisher = False)
    Skewness = skew(data)
    Mean     = np.mean(data)
    Median   = np.median(data)
    
    ''' Spectral mean in different freq. bands '''
    Spectral_mean = 1 / (freq_ix['Beta'][-1] - freq_ix['Delta'][0]) * (Pow_Delta + 
            Pow_Theta + Pow_Alpha + Pow_Beta + Pow_Sigma + Pow_Sigma_slow + Pow_Sigma_fast) 
    
    feat = [pow_total, Pow_Delta, Pow_Theta, Pow_Alpha, Pow_Beta, Pow_Sigma, cA_mean[0], cA_std[0],
            cA_Energy[0], cD_Energy[0],  cD_mean[0], cD_std[0],  Entropy_D[0], Entropy_A[0],
            Pow_Sigma_slow, Pow_Sigma_fast, Kurt, Skewness, Mean, Median, Spectral_mean]
    
    Features = np.row_stack((Features,feat))
    
#%% Replace the NaN values of features with the mean of each feature column

aa, bb = np.where(np.isnan(Features))
for j in np.arange(int(len(aa))):
    Features[aa[j],bb[j]] = np.nanmean(Features[:,bb[j]])
    
#%% Normalizing features
Feat_train = Features[:int(len(X_train)),:]
Feat_test = Features[int(len(X_train)):,:]
sc = StandardScaler()
Feat_train = sc.fit_transform(Feat_train)
Feat_test = sc.transform(Feat_test)

#%% Shuffle train and test data with rand perumtation
rp_train = np.random.permutation(len(Feat_train))
rp_test = np.random.permutation(len(Feat_test))

Feat_train_rp = Feat_train[rp_train,:]
Feat_test_rp  = Feat_test[rp_test,:]
y_train_rp    = ytrain[rp_train,1]
y_test_rp     = ytest[rp_test,1]

#%% Random Forest
classifier = RandomForestClassifier(n_estimators = 500)
#classifier.fit(Feat_train_rp, y_train_rp)

#y_pred = classifier.predict(Feat_test_rp)


#cm = confusion_matrix(y_test_rp, y_pred)
#Acc = (cm[0,0]+cm[1,1]) / (cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

#%% PLOTTING
#FeatureImportance = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
#sb.barplot(y=FeatureImportance, x=FeatureImportance.index)
#plt.show()


#%% CROSS-VALIDATION
X    = np.row_stack((Feat_train_rp, Feat_test_rp))
y    = np.concatenate((y_train_rp, y_test_rp))
#RP   = np.random.permutation(len(X))
#X_RP = X[RP,:]
#Y_RP = Y[RP,:]

accuracies = cross_val_score(estimator = classifier, X = X, 
                             y = y, cv = 10)
Acc_cv10 = accuracies.mean()
std_cv10 = accuracies.std()

#%% Grid search
# DEFINING PARAMATERS
# Number of trees in random forest
n_estimators = [int(x) for x in np.arange(10, 2000, 30)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.arange(10, 110, 30)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
t = time.time() # tic
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion' :['gini', 'entropy']}

grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2)

print('Total configurations to be checked: ',
      len(n_estimators) * len(max_features) * len(max_depth) *
      len(min_samples_split) * len(min_samples_leaf) * len(bootstrap))

grid_search.fit(Feat_train_rp, y_train_rp)
BestParams_GridSearch = grid_search.best_params_
Bestscore_GridSearch  = grid_search.best_score_

elapsed = time.time() - t # toc

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

