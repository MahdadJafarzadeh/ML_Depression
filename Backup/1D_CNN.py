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

# Load train and test splits
tic = time.time() 
fname = ('P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/tr80_fp1fp2.h5')
with h5py.File(fname, 'r') as rf:
    x_test_fp1 = rf['.']['x_test_fp1'].value
    x_train_fp1 = rf['.']['x_train_fp1'].value
    y_test_fp1 = rf['.']['y_test_fp1'].value
    y_train_fp1 = rf['.']['y_train_fp1'].value
print('train and test data loaded in : {} secs'.format(time.time()-tic))

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy


