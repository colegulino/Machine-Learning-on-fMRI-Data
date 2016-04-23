# -*- coding: utf-8 -*-
"""
Test Which iteration and feature selection is best
"""

# Imports
import numpy as np
import scipy.io as sio
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

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

# Scale the training data first
X_scale = preprocessing.scale( Xtrain )
data = []

for i in xrange(15,16):
    for j in xrange( 120, 121 ):
        # Use PCA to reduce the dimensionality
        comp =  i # number of components
        cw = {}
        cw[1] = 1
        cw[0] = 1
        cw[3] = 1
        it = j
        pca = decomposition.PCA(n_components=comp)
        # Set up the classifier
        clf = SVC(C=50.0, cache_size=200, class_weight=cw, coef0=0.0,
                  degree=3, kernel='rbf', max_iter=-1, 
                  probability=True,random_state=None, shrinking=True, 
                  tol=0.01, verbose=False)
        
        # Setup the pipeline
        pipe = Pipeline([('pca', pca), ('svm', clf)])
        pipe.fit(X_scale, np.ravel(Ytrain))
        
        
        # Run cross validation
        scores = cross_validation.cross_val_score(clf, X_scale, np.ravel(Ytrain), cv=10)
        data = data + [i, j, scores.mean()]


        
        
        
        


