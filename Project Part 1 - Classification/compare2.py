# -*- coding: utf-8 -*-
"""
File to compare multiple classifiers
"""

import numpy as np

gnb = np.genfromtxt('gnb.csv', delimiter=',')
svm = np.genfromtxt('svm.csv', delimiter=',')

gnb = np.array(gnb)
svm = np.array(svm)

new = np.zeros((1001,3))

for i in xrange(0, 1001):
    if( max(svm[i,:]) > max(gnb[i,:])):
        new[i,:] = svm[i,:]
    else:
        new[i,:] = gnb[i,:]
np.savetxt('prediction.csv', new, delimiter=",")
