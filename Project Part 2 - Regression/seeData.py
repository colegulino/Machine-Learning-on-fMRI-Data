# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:45:14 2015

@author: Cole Gulino
"""

# Imports
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import preprocessing

# Import your data
events_1000 = sio.loadmat('Data/events_1000.mat')
missIdx = sio.loadmat('Data/missIdx.mat')
provideData_1000 = sio.loadmat('Data/provideData_1000.mat')
provideIdx = sio.loadmat('Data/provideIdx.mat')
trainData = sio.loadmat('Data/Train.mat')
testData = sio.loadmat('Data/Test.mat')

events = np.array(events_1000.get('events'))
missidx = np.array(missIdx.get('missIdx'))
provideData = np.array(provideData_1000.get('provideData'))
provideidx = np.array(provideIdx.get('provideIdx'))

Xtrain = np.array(trainData.get('Xtrain'))
Ytrain = np.array(trainData.get('Ytrain'))
Xtest = np.array(testData.get('Xtest'))


# Scale the Data
Xtrain = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Xtrain)

# Get the principal components
n_comp = 50
pca = PCA(n_components=n_comp)
pca.fit(Xtrain)
# Get component vectors
comp = pca.components_

# Scale the provided test data
provideData = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(provideData)
# Run the data through pca
provideDataPCA = pca.fit_transform(provideData)

# Multiply the components and the new data in order to reconstruct
reconstructX = np.around(provideDataPCA.dot(comp) + np.mean(provideDataPCA))

# Put into a csv file
np.savetxt('prediction.csv', np.around(reconstructX, decimals=3), delimiter=",")