# -*- coding: utf-8 -*-
"""
Get the PCA information, so that we can see which of the features contributes
most to the variance
"""

'''
Get the data first
'''
# Imports
import scipy.io as sio
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.pipeline import Pipeline
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

svm = SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
          degree=3, gamma=100, kernel='rbf', max_iter=-1, 
          probability=True,random_state=None, shrinking=True, 
          tol=0.001, verbose=False)

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
Xnew = pca.fit(Xtrain)

'''
Plot the PCA spectrum
'''
plt.figure(1, figsize=(4,3))
plt.clf()
plt.axes([0.2, 0.2, 0.7, 0.7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

