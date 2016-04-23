# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:25:09 2015

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
# Get events data and indices
events = events_1000.get('events')
missidx = missIdx.get('missIdx')
provideData = provideData_1000.get('provideData')
provideidx = provideIdx.get('provideIdx')
# Get training data
Xtrain = trainData.get('Xtrain')
Ytrain = trainData.get('Ytrain')
Xtest = testData.get('Xtest')

'''
Get full training data
'''
# Concatenate the Xtrain and Xtest
x = np.vstack((Xtrain, Xtest))
Xtrain = x[:, provideidx[0,0]-1]
Ytrain = x[:, missidx[0,0]-1]

for i in xrange(1, np.shape(x)[1] + 1):
    #print i
    if( i in provideidx and i != provideidx[0,0] ):
        Xtrain = np.vstack(( Xtrain, x[:,i-1] ))
    elif( i in missidx and i != missidx[0,0] ):
        Ytrain = np.vstack(( Ytrain, x[:,i-1] ))

Xtrain = Xtrain.T
Ytrain = Ytrain.T
'''
Set up dimensionality reduction
'''
# Parameters
comp = 453
k_ = 180
percentile_ = 4
# PCA
pca = decomposition.PCA(n_components=comp)
# Feature selection
selection = SelectKBest(k=k_)
# Percentile selection
class_stuff = SelectPercentile(f_classif, percentile=percentile_)
# Feature Union
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection),("class_stuff",class_stuff)])
'''
Reduce dimensions of Xtrain
'''
# Scale Xtrain
Xtrain = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Xtrain)
# Reduce dimensions
Xtrain = pca.fit_transform(Xtrain)
'''
Set up and use linear regression
'''
# Set up linear regression
lr = linear_model.LinearRegression(fit_intercept=True,
                                   normalize=True)
# Train the model using the training set
lr.fit(Xtrain, Ytrain)
# Get the RMSE score 
Ycv = lr.predict(Xtrain)
print "Linear Regression: "
print np.mean(cross_validation.cross_val_score(lr, Xtrain, Ytrain, scoring="mean_squared_error", cv=10))
