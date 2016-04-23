# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 02:22:14 2015

@author: cole
"""

# Imports
import numpy as np
import scipy.io as sio
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation

'''
Import your data 
'''
# Load the .mat files
events_1000 = sio.loadmat('Data/events_1000.mat')
missIdx = sio.loadmat('Data/missIdx.mat')
provideData_1000 = sio.loadmat('Data/provideData_1000.mat')
provideIdx = sio.loadmat('Data/provideIdx.mat')
trainData = sio.loadmat('Data/Train.mat')
testData = sio.loadmat('Data/Test.mat')
yTest = sio.loadmat('Data/Ytest.mat')
# 
events = events_1000.get('events')
missidx = missIdx.get('missIdx')
provideData = provideData_1000.get('provideData')
provideidx = provideIdx.get('provideIdx')

Xtrain = trainData.get('Xtrain')
Ytrain = trainData.get('Ytrain')
Yt2 = yTest.get('Ytest')
Xtest = testData.get('Xtest')
missingData = np.genfromtxt ('Data/prediction.csv', delimiter=",")

'''
Get unlabeled training data
'''
# Preallocate size of unlabeled data
X2 = np.zeros((1000, 5903))
# Get fully construct unlabeled data
# Do missing data first
i = 0
for index in xrange(0, np.shape(missidx)[1]):
    #print missidx[0][index]
    X2[:, missidx[0][index]-1] = missingData[:,i]
    i = i + 1
# Now do for provided data
i = 0
for index in xrange(0, np.shape(provideidx)[1]):
    X2[:, provideidx[0][index]-1] = provideData[:, i]
    i = i + 1
    
# Add this data to the training data from part 1
X = np.vstack((Xtrain, X2))
print np.shape(X)
# Get the Y for the semi-supervised learning
Y = np.zeros((1501,1))
Y[:501,:] = Ytrain
Y[501:,:] = -1