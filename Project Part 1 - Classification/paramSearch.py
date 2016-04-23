# -*- coding: utf-8 -*-
"""
Parameter grid search
"""
# Imports
import numpy as np
import scipy.io as sio
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

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


for i in xrange(23, 24):
    # Set up the pipeline
    svm = SVC()
    c = np.r_[10:100]
    g = np.r_[100:500]
    param_grid = dict(gamma=g, C=c)
    
    # Use PCA on the data
    Xtrain = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Xtrain)
    comp = 450
    pca = decomposition.PCA(n_components = comp)
    pca.fit(Xtrain)
    Xpca = pca.transform(Xtrain)
    
    # Now scale the data from PCA
    X_scale = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Xpca)
    
    # Now set up the classifier
    clf = SVC(C=10.0, cache_size=200, kernel='rbf', 
              max_iter=-1, probability=True,random_state=None, 
              shrinking=True, gamma = 0.0001,
              tol=0.001, verbose=False)
    grid_search = GridSearchCV(svm, param_grid)
    grid_search.fit(X_scale, np.ravel(Ytrain))
    
    print "components: "
    print i
    print "Best Score: "
    print grid_search.best_score_
    
    bp = grid_search.best_estimator_.get_params()
    
    for param_name in sorted(param_grid.keys()):
        print param_name + " " + str(bp[param_name])
