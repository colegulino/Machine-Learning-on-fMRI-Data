# -*- coding: utf-8 -*-
"""
Multi-Class SVM without using PCA to reduce dimensionality
"""

# Imports
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn import preprocessing

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

testX = np.array( testX, np.float32 )

# Set up the classifier
clf = SVC(C=50.0, cache_size=2000, class_weight=None, coef0=0.0,
          degree=3, gamma=100, kernel='rbf', max_iter=75, 
          probability=True,random_state=None, shrinking=True, 
          tol=0.0000001, verbose=False)

# Scale and normalize the data
Xscale = preprocessing.scale(Xtrain)
Xnorm = preprocessing.normalize( Xscale, norm='l2')
# Train the classifier
clf.fit( Xnorm, np.ravel(Ytrain) )

# Run the values on the test set
testX = testData.get('Xtest')
testScale = preprocessing.scale(testX)
testNorm = preprocessing.normalize(testScale, norm='l2')
testY = []
tesY = np.array(testY, np.float32)
prob = []
prob = np.array(prob, np.float32)
# Get the class prediction
testY = clf.predict( testNorm )
# Get the probabilities
prob = clf.predict_proba( testNorm )