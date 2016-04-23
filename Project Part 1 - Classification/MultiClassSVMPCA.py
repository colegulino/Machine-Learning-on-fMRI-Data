# -*- coding: utf-8 -*-
"""
Multi-Class SVM without using PCA to reduce dimensionality
"""

# Imports
import numpy as np
import scipy.io as sio
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation

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

# Set up the classifier
# Use PCA to reduce the dimensionality
comp = 400 # number of components
cw = {}
cw[1] = 1
cw[0] = 1
cw[3] = 1
pca = decomposition.PCA(n_components=comp)
# Set up the classifier
clf = SVC(C=10.0, cache_size=200, class_weight=cw, coef0=0.0,
          degree=3, kernel='rbf', max_iter=-1, 
          probability=True,random_state=None, shrinking=True, 
          tol=0.01, verbose=False)

X_scale = preprocessing.scale( Xtrain )
X_norm = preprocessing.normalize(X_scale, norm='l1')
pca.fit(X_norm)
X_pca = pca.transform(X_norm)
X_scale2 = preprocessing.scale(X_pca)
X_norm2 = preprocessing.normalize(X_scale2, norm='l1')
clf.fit( X_norm2, np.ravel(Ytrain))
# Run cross validation
scores = cross_validation.cross_val_score(clf, X_norm2, np.ravel(Ytrain), cv=10)
print scores.mean()

# Run the values on the test set
testX = testData.get('Xtest')
testY = []
testY = np.array(testY, np.float32)
prob = []
prob = np.array(prob, np.float32)

# Use PCA on the test set
testScale = preprocessing.scale( testX )
testScale = preprocessing.normalize(testScale, norm='l1')
#pca.fit(testScale)
testNew = pca.transform(testScale)
testNew_scale = preprocessing.scale(testNew)
testNorm = preprocessing.normalize( testNew, norm='l1')

# Get the class prediction
testY = clf.predict( testNorm )
# Get the probabilities
prob = clf.predict_proba( testNorm )

# Put into a csv file
np.savetxt('prediction.csv', prob, delimiter=",")
