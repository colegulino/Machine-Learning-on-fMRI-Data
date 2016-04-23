# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:36:34 2015

@author: colegulino
"""
import numpy as np
import scipy.io as sio
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation
from sklearn.naive_bayes import  GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile
from skimage import filters
import math

# Get distance
def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2)+pow(z1-z2,2))
# Do Spatial Smoothing on data
def spatialSmoothing(X, x, y, z, gamma, sigma):
    # loop through the collumns of the data for each row
    newX = X
    for i in range(0, 501): # Do the rows
        for j in range(0, 5903):
            trans = []
            # insert the current voxel into both lists
            trans.append(X[i,j])
            x1 = x[j]
            y1 = y[j]
            z1 = z[j]
            for k in range(0, 5903):
                x2 = x[k]
                y2 = y[k]
                z2 = z[k]
                dist =[]
                dist.append(0)
                # Decide if the x,y,z components of the thing is close enough
                if(k != j and dist(x1,y1,z1,x2,y2,z2) <= gamma):
                    trans.append(X[i,k])
                    dist.append(dist(x1,y1,z1,x2,y2,z2))
            # Now go through and put the order of the array as distance to the first
            for i in range(0, len(x)):
                for j in range(i+1, len(x)):
                    if( dist[j] < dist[i] ):
                        dist[j], dist[i] = dist[i], dist[j]
                        trans[j], trans[i] = trans[i], trans[j]
            if( len(trans) > 1 ):
                filterTrans = filters.gaussian_filter(trans, sigma, output=True)
            # First value of filterTrans is what you want to keep
            # Replace it with the previous value in the newX
            newX[i,j] = filterTrans[0]
    return newX
            
            
# Import the data
trainData = sio.loadmat('Train.mat')
testData = sio.loadmat('Test.mat')

# Get the values of the train data
Xtrain = trainData.get('Xtrain')
Ytrain = trainData.get('Ytrain')
eventsTrain = trainData.get('eventsTrain')
subjectsTrain = trainData.get('subjectsTrain')
x = trainData.get('x')
y = trainData.get('y')
z = trainData.get('z')

# Get the values of the test data
# Get the test data into a numpy array
testX = testData.get('Xtest')
eventsTest = testData.get('eventsTest')
subjectsTest = testData.get('subjectsTest')
testX = np.array( testX, np.float32)