#!/usr/bin/env python
'''
An implementation of Multiclass Winnow. Uses a set of weights for each of the classes. Accepts only binary features.
'''

import numpy as np
from scipy.linalg.basic import inv
from LogReader import LogReader
import math as math


class binaryWinnowvar:

    def __init__(self, features, classes,params):
        # Assert that features are binary
        #assert(np.array_equal(np.unique(features),np.array([0,1])))
        self.trainX = features
        self.trainY = classes
        self.class_names = np.unique(classes)
        self.num_features = self.trainX.shape[1]
        self.num_classes = 2



        self.thresh = params[0]
        # those could be used as parameters
        self.nu = params[1]

    def print_info(self):

        print "[BinaryWinnow] Number of observations: ", self.trainY.shape[0]
        print "[BinaryWinnow] Number of features: ", self.num_features
        print "[BinaryWinnow] Number of classes: ", self.num_classes
        print "[BinaryWinnow] Promotion parameter: ", self.p
        print "[BinaryWinnow] Demotion parameter: ", self.d
        # add function to print info

    def train(self):

        #Initialize all the weights to 1
        self.weights = np.ones([self.num_features, 1])
        # Track number of mistakes
        err_num = 0

        for index in range(self.trainX.shape[0]):
            prediction = np.dot(self.trainX[index, :], self.weights)
            if prediction < self.thresh:
                predicted_label = -1
            else:
                predicted_label = 1

            if not(predicted_label == self.trainY[index]):
                err_num += 1
                #print "predicted", predicted_label
                self.update_weights(self.trainX[index,:],self.trainY[index])

        print "[BinaryWinnow] Number of mistakes at learning: ", err_num
        print "[BinaryWinnow] Learned weights \n",self.weights

    def predict(self, data_pt):

        return np.dot(data_pt, self.weights)

    def update_weights(self,data_pt,true_label):
        # for the wrongly labeled class, find the features ==1, and demote or promote their weights depending on type
        # of misclassification
        tmp = self.weights.T*np.exp(self.nu*true_label*data_pt)
        self.weights = tmp.T

    def test(self, data_pt, true_label):

        if self.predict(data_pt) >= self.thresh and true_label == 1:
            return True
        elif self.predict(data_pt) < self.thresh and true_label == -1:
            return True
        else:
            return False



if __name__ == "__main__":

    print "in binaryWinnowvar"

