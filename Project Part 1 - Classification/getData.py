# -*- coding: utf-8 -*-
"""
Load and analyze data 
"""

# Imports
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import matplotlib.pyplot as plt

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

# Get the x and y for each class
zeroX = []
zeroY = []
oneX = []
oneY = []
threeX = []
threeY = []
for i in xrange(0, len(Xtrain)):
    if( Ytrain[i] == 1 ):
        oneX.append( Xtrain[i] )
        oneY.append( Ytrain[i] )
    if( Ytrain[i] == 0 ):
        zeroX.append( Xtrain[i] )
        zeroY.append( Ytrain[i] )
    if( Ytrain[i] == 3 ):
        threeX.append( Xtrain[i] )
        threeY.append( Ytrain[i] )
zeroOneX = zeroX + oneX
zeroOneY = zeroY + oneY
zeroThreeX = zeroX + threeX
zeroThreeY = zeroY + threeY
oneThreeX = oneX + threeX
oneThreeY = oneY + threeY

# Delete those you do not need
del zeroX, zeroY, oneX, oneY, threeX, threeY
      
# Convert the XTrain and YTrain to numpy arrays
zeroOneX = np.array(zeroOneX, np.float32)
zeroOneY = np.array(zeroOneY, np.float32)
zeroThreeX = np.array(zeroThreeX, np.float32)
zeroThreeY = np.array(zeroThreeY, np.float32)
oneThreeX = np.array(oneThreeX, np.float32)
oneThreeY = np.array(oneThreeY, np.float32)

