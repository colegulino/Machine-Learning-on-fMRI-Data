# -*- coding: utf-8 -*-
"""
SVM again with new parameters
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

# Use PCA on the data
Xtrain = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Xtrain)
comp = 450
pca = decomposition.PCA(n_components = comp)
pca.fit(Xtrain)
pca.fit(testX)
Xpca = pca.transform(Xtrain)

# Now scale the data from PCA
X_scale = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Xpca)

# Now set up the classifier
clf = SVC(C=10.0, cache_size=200, kernel='rbf', 
          max_iter=-1, probability=True,random_state=None, 
          shrinking=True, gamma = 0.0001,
          tol=0.001, verbose=False)
         
# Now train the classifier
clf.fit(Xpca, np.ravel(Ytrain))

# Run cross validation
scores = cross_validation.cross_val_score(clf, Xpca, np.ravel(Ytrain), cv=10)
print scores.mean()

# Now run the test set

testX = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(testX)
# Run pca on the test set
testPCA = pca.transform(testX)

# Scale the test data from PCA
testScale = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(testPCA)

# Now predict
# Get the class prediction
testY = clf.predict( testPCA )
# Get the probabilities
prob = clf.predict_proba( testPCA )

new = np.zeros((1001,3))

for i in xrange(0, 1001):
    if( testY[i]== 0):
        new[i,:] = [1.0000, 0.0000, 0.0000]
    elif( testY[i] == 1):
        new[i,:] = [0.0000, 1.0000, 0.0000]
    else:
        new[i,:] = [0.0000, 0.0000, 1.0000]

# Put into a csv file
np.savetxt('prediction2.csv', new, delimiter=",")


