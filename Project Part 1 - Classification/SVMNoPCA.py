# -*- coding: utf-8 -*-
"""
SVM Implementation without using any PCA to lower the dimensions
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

# Use SVM with RGF Gaussian Kernel from scikit-learn
# Set up three one-one classification algorithms
# oneTwo - classifies those that have class one and two only
# oneThree - classifies those that have class one and three only
# twoThree - classifies those that have class two and three only
clfZeroOne = SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
          degree=3, gamma=100, kernel='rbf', max_iter=-1, 
          probability=False,random_state=None, shrinking=True, 
          tol=0.001, verbose=False)

clfZeroThree = SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
           degree=3, gamma=100, kernel='rbf', max_iter=-1,
           probability=False,random_state=None, shrinking=True, 
           tol=0.001, verbose=False)

clfOneThree = SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
           degree=3, gamma=100, kernel='rbf', max_iter=-1, 
           probability=False,random_state=None, shrinking=True, 
           tol=0.001, verbose=False)

 
# Train the classifiers
np.reshape(zeroOneX, (275, 5903))
np.reshape(zeroThreeX, (378, 5903))
np.reshape(oneThreeX, (349, 5903))
clfZeroOne.fit( zeroOneX, np.ravel(zeroOneY) )
clfZeroThree.fit( zeroThreeX, np.ravel(zeroThreeY) )
clfOneThree.fit( zeroThreeX, np.ravel(zeroThreeY) )


# Get the test data into a numpy array
testX = testData.get('Xtest')

testX = np.array( testX, np.float32)
testY01 = []
testY01 = np.array( testY01, np.float32)
testY03 = []
testY03 = np.array( testY03, np.float32)
testY13 = []
testY13 = np.array( testY13, np.float32)

# Get the test predictions
testY01 = clfZeroOne.predict( testX )
testY03 = clfZeroThree.predict( testX )
testY13 = clfOneThree.predict( testX )






