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
import h5py
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.optimizers import adam
from keras.constraints import maxnorm

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

# Standardization
sc = StandardScaler()
# First fit StandardScaler into one epoch of data
trainX[0,:,:] = sc.fit_transform(trainX[0,:,:])
# Then apply this transfromation to all the remaining train and test epochs
for i in np.arange(1,len(trainX)):
    trainX[i,:,:] = sc.transform(trainX[i,:,:])
    
for i in np.arange(0,len(testX)):
    testX[i,:,:] = sc.transform(testX[i,:,:])

# Shuffle train and test data with rand perumtation
rp_train = np.random.permutation(len(trainy))
rp_test = np.random.permutation(len(testy))

trainX = trainX[rp_train,:,:]
trainy = trainy[rp_train,:]
testX  = testX[rp_test,:,:]
testy  = testy[rp_test,:]


def create_model(learn_rate = .01, momentum = 0, weight_constraint = 0, dropout_rate = 0.0, neurons =50):
    verbose = 1
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size= 5, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = 32, kernel_size= 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = 32, kernel_size= 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides = 2))

    model.add(Flatten())
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))
    optimizer = adam(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

########## GRID SEARCH #######
tic = time.time() 

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size        = [5, 10, 20, 40, 60]
epochs            = [5, 10]
learn_rate        = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum          = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons           = [1, 5, 10, 15, 20, 25, 30]

# Create grid search
param_grid = dict(batch_size=batch_size, epochs=epochs, learn_rate=learn_rate,
                  momentum=momentum, dropout_rate=dropout_rate, 
                  weight_constraint=weight_constraint, neurons = neurons)
grid = GridSearchCV(estimator=model,n_jobs = -1, param_grid=param_grid, cv=3)
grid_result = grid.fit(trainX, trainy)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


print('Total time for grid search : {} secs'.format(time.time()-tic))

'''    
# fit and evaluate a model
verbose, epochs, batch_size = 1, 100, 1
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
model = Sequential()
model.add(Conv1D(filters=64, kernel_size= 5, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(MaxPooling1D(pool_size=2, strides = 2))
model.add(Conv1D(filters=64, kernel_size= 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2))
model.add(Conv1D(filters=128, kernel_size= 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2))
model.add(Conv1D(filters=128, kernel_size= 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2))
model.add(Conv1D(filters=128, kernel_size= 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
tic = time.time() 
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
print('time to fit training set to the model : {} secs'.format(time.time()-tic))

# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
 '''
