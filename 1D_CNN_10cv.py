# -*- coding: utf-8 -*-

"""
1D Convolutional Neural Networks
Created: 2020/03/18

Script to classify deprressed from normal people suing single channel EEG data
during REM sleep.

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from scipy.signal import butter, lfilter
import h5py
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

 
# Load train and test splits
tic = time.time() 
fname = ('C:/Users/mahda/Documents/Python Machine Learning/MachineLearning/Machine Learning A-Z Template Folder/Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)/CNN_1D/tr80_fp1fp2.h5')
with h5py.File(fname, 'r') as rf:
    x_test_fp1 = rf['.']['x_test_fp1'].value
    x_train_fp1 = rf['.']['x_train_fp1'].value
    y_test_fp1 = rf['.']['y_test_fp1'].value
    y_train_fp1 = rf['.']['y_train_fp1'].value
print('train and test data loaded in : {} secs'.format(time.time()-tic))

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
trainX = x_train_fp1.reshape((x_train_fp1.shape[0], x_train_fp1.shape[1], n_features))
testX = x_test_fp1.reshape((x_test_fp1.shape[0], x_test_fp1.shape[1], n_features))
trainy = y_train_fp1
testy = y_test_fp1

X = np.concatenate((trainX, testX))
Y = np.concatenate((trainy, testy))

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b,a)
    y = lfilter(b, a, data)
    return y

# Apply filter
X = butter_bandpass_filter(data=X, lowcut=.3, highcut=20, fs=200, order=2)

# Standardization
sc = StandardScaler()
# First fit StandardScaler into one epoch of data
X = X.flatten()
X = X.reshape(1,-1).transpose()
X = sc.fit_transform(X)

X = X.reshape(int(len(X)/6000),6000,1)

# Shuffle train and test data with rand perumtation
rp = np.random.permutation(len(Y))

X = X[rp,:,:]
Y = Y[rp,:]

#%% Plotting
#plt.plot(X)   
#%%
def create_model():
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    model = Sequential()
    model.add(Conv1D(filters = 5, kernel_size= 5, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = 5, kernel_size= 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = 10, kernel_size= 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = 10, kernel_size= 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = 15, kernel_size= 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


tic = time.time() 
model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=10, verbose=1)
# evaluate using 10-fold cross validation
results = cross_val_score(model, X, Y, cv=10)
print(results.mean())

print('Taken time : {} secs'.format(time.time()-tic))
