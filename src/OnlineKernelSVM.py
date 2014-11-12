#!/usr/bin/env python 
'''
An implementation of Kernel SVM with training using online gradient descent
'''

import numpy as np
from scipy.linalg.basic import inv
import pdb
from LogReader import LogReader
import pickle

class OKSVM:
    
    def __init__(self, features, classes, params):
        #assert(len(params) == 3)
        self.trainX = features
        self.trainY = classes
        
        self.supportVecs = np.zeros((1, self.trainX.shape[1]))
        self.alphas = np.zeros(1)
        self.learning_rate = params[0]
        self.kernel_type = params[1]
        self.regularizer = params[2] 
        
    def train(self):
        for data_pt in range(self.trainX.shape[0]):
#             print 'alphas', self.alphas
            prediction = self.eval_function(self.alphas, self.supportVecs, self.trainX[data_pt])
#             print 'kernelResult' , kernelResult
#             print 'label', self.trainY[data_pt]
            err =  self.eval_loss(prediction, self.trainY[data_pt])
#             print 'err', err
            print data_pt
            if err > 0:
#                 self.supportVecs = np.concatenate((self.supportVecs.T, np.array([self.trainX[data_pt]]))).T
                
                #Add current data point to the 'supportVecs'
                self.supportVecs = np.vstack( (self.supportVecs, self.trainX[data_pt]) )
#                 print 'alphas', self.alphas
                '''
                update = {  add kernel wighted by \eta y_t if correct
                         {  shrink all other weights by 1 - 2 \eta \lambda  
                '''
                self.alphas = self.alphas * (1.0 - 2*self.learning_rate*self.regularizer)
#                 print 'supportVecs', self.supportVecs
                self.alphas = np.append(self.alphas, self.learning_rate*self.trainY[data_pt])


    def eval_loss(self, prediction, label_value):
        '''
        ||f||^2 = <f, f> = 
        '''
        #Commenting this out until we have a smarter way of removing support vectors
#         K = np.zeros((self.supportVecs.shape[0], self.supportVecs.shape[0]) )
#         for index,vector in enumerate(self.supportVecs):
#             K[index,:] = self.linear_kernel(self.supportVecs, vector) 
        return max(0.0, 1 - (prediction * label_value)) #+ self.regularizer*np.dot(self.alphas.T, np.dot(K, self.alphas)) )
        
    def eval_function(self, alphas, supportvecs, dataX):
#         s = 0
#         print 'supportvecs shape', supportvecs.shape
#         for x in range(supportvecs.shape[1]):
#             #for y in range(dataX.shape[0]):
#                 # rbf
#                 #print x
#                 #print y
#                 #print alphas[x]
#             diff = dataX - supportvecs[:,x]
#             s += alphas[x] * np.exp(-np.linalg.norm(diff) / self.sigma)
# 
#         return s
        if self.kernel_type == 'linear':
            return np.sum(np.dot(alphas, self.linear_kernel(supportvecs, dataX)) )
        if self.kernel_type == 'rbf':
            return np.sum(np.dot(alphas, self.rbf_kernel(supportvecs, dataX)))
        assert(0)
    
    def linear_kernel(self, supportvecs, dataX):
        return np.dot(supportvecs, dataX)
    
    def rbf_kernel(self, supportvecs, dataX):
        #todo
        assert(0)
        return 0
    
    def predict(self, dataX):
        #Check if sane data point
        assert(dataX.shape == self.trainX[0].shape)
        return self.eval_function(self.alphas, self.supportVecs, dataX)
        
    def test(self, dataX, true_label):
        if self.predict(dataX)*true_label > 0:
            return True
        else:
            return False
    
    def linear_feature(self, dataX):
        return np.append(dataX, 1.0)


if __name__ == "__main__":
    #Test the regression
    Xs = np.array([[1,1],[-1,-1], [1,-1], [-1,1]])
    Ys = np.array([1, -1, -1, 1] )

    svm_params = [0.4, 'linear', 0.01]
    test_svm = OKSVM(Xs, Ys, svm_params)
    test_svm.train()
    print test_svm.test(np.array([0, 1]) , 1)
    print test_svm.test(np.array([1, -0.1]) , -1)
    print test_svm.test(np.array([0, -1]), -1 )
    print test_svm.test(np.array([-1, 0.1]), 1 )
    
    train_log_object = LogReader('../data/oakland_part3_am_rf.node_features')
    test_log_object = LogReader('../data/oakland_part3_an_rf.node_features')
    train_points = train_log_object.read()
    test_points = test_log_object.read()
    trainXs = np.array([point._feature for point in train_points])
    trainYs = np.array([point._label for point in train_points])
    testXs = np.array([point._feature for point in test_points])
    testYs = np.array([point._label for point in test_points])
    test_svm = OKSVM(trainXs, trainYs, svm_params)
    test_svm.train()
    
    evals = []
    for index in range(len(testXs)):
        dataX = testXs[index]
        true_label = testYs[index]
        classifier_index = test_svm.predict(dataX)
        evals.append(test_svm.test(dataX, true_label))
    #Now evaluate accuracy
    #True == 1, thus sum
    print '[OneVsAll] Accuracy = ', float(sum(evals))/len(evals)
    print 'done'
    #print test_regression.predict(np.array([0.5, 1])), test_regression.test(np.array([0.5, 1]), 1)