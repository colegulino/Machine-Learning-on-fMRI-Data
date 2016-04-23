# -*- coding: utf-8 -*-
"""
Using 
"""
# Imports
import numpy as np
import scipy.io as sio
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, SelectFromModel
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
comp = 453 # number of components
cw = {}
cw[1] = 1
cw[0] = 1
cw[3] = 1
pca = decomposition.PCA(n_components=comp)

selection = SelectKBest(k=180)
class_stuff = SelectPercentile(f_classif, percentile = 4)

# Set up the classifier
clf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                             max_depth=None, min_samples_split=1,
                             oob_score=False)

Xtrain = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Xtrain)
tree_selection = SelectFromModel(clf, prefit=False)
combined_features = FeatureUnion([("pca", pca),
                                ("model_fit", tree_selection)])
X_pca = combined_features.fit(Xtrain, np.ravel(Ytrain)).transform(Xtrain)

clf.fit( X_pca, np.ravel(Ytrain))


# Run cross validation
scores = cross_validation.cross_val_score(clf, X_pca, np.ravel(Ytrain), cv=10)
print(scores.mean())


# Run the values on the test set
testX = testData.get('Xtest')
testY = []
testY = np.array(testY, np.float32)
prob = []
prob = np.array(prob, np.float32)

# Use PCA on the test set
testX = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(testX)
testNew =  combined_features.transform(testX)

# Get the class prediction
testY = clf.predict( testNew )
# Get the probabilities
prob = clf.predict_proba( testNew )

# convert to labels

new = np.zeros((1001,3))

for i in range(0, 1001):
    if( prob[i,0] == max(prob[i,:])):
        new[i,:] = [1.0000, 0.0000, 0.0000]
    elif( prob[i,1] == max(prob[i,:])):
        new[i,:] = [0.0000, 1.0000, 0.0000]
    else:
        new[i,:] = [0.0000, 0.0000, 1.0000]


# Put into a csv file
np.savetxt('prediction.csv', new, delimiter=",")