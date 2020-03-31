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
    
for i in np.arange(1,len(testX)):
    testX[i,:,:] = sc.transform(testX[i,:,:])

# Shuffle train and test data with rand perumtation
rp_train = np.random.permutation(len(trainy))
rp_test = np.random.permutation(len(testy))

trainX = trainX[rp_train,:,:]
trainy = trainy[rp_train,:]
testX  = testX[rp_test,:,:]
testy  = testy[rp_test,:]


def evaluate_model(trainX, trainy, testX, testy, n_filters):
    verbose, epochs, batch_size = 1, 10, 5
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters = n_filters, kernel_size= 5, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = n_filters, kernel_size= 5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = n_filters, kernel_size= 5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = n_filters, kernel_size= 5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters = n_filters, kernel_size= 5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# summarize scores
def summarize_results(scores, params):
	print(scores, params)
	# summarize mean and standard deviation
	for i in range(len(scores)):
		m, s = np.mean(scores[i]), np.std(scores[i])
		print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
	# boxplot of scores
	plt.boxplot(scores, labels=params)
	plt.savefig('exp_cnn_filters.png')

# run an experiment
def run_experiment(params, repeats=10):

	# test each parameter
	all_scores = list()
	for p in params:
		# repeat experiment
		scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%d #%d: %.3f' % (p, r+1, score))
			scores.append(score)
		all_scores.append(scores)
	# summarize results
	summarize_results(all_scores, params)

# run the experiment
n_params = [8, 16, 32, 64, 128, 256]
run_experiment(n_params) 
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

# creating log file 
a = np.random.rand(5)
b = np.random.rand(15)
c = np.random.rand(2,7)

File_object = open(r'Log_file1.txt','a+')
File_object.write("This is a: \n " + str(a) + '\n')
File_object.write("This is b: \n " + str(b)+ '\n')
File_object.write("This is c: \n " + str(c)+ '\n')
File_object.close()