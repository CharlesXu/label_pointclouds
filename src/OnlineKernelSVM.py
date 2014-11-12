#!/usr/bin/env python 
'''
An implementation of Kernel SVM with training using online gradient descent
'''

import numpy as np
from scipy.linalg.basic import inv

class OKSVM:
    
    def __init__(self, features, classes, params):
        #assert(len(params) == 3)
        self.trainX = features
        self.trainY = classes
        
        #self.observation_error_sigma = params[0]
        #self.prior_mean = params[1]*np.ones(self.trainX.shape[1]);
        #self.prior_sigma = params[2];

        self.supportVecs = np.zeros((self.trainX.shape[1], 0))
        self.weights = np.zeros((0))
        self.sigma = 0.2
        
    def train(self):
        '''
        The MAP estimate of p(w| D) is the mean of the posterior (which is a gaussian distribution)
        For prediction, we need p(y| x, D)
        
        p(y | x, D) = \int p(y|x, D, w) p(w|x, D) dw
                    = \int p(y|x, w) p(w|D) 
                    = \int N(y|w^Tx, a^-1) N(w | \mu, \lambda^-1)         
                    = N( y| \mu_w^T x , x^T \Sigma_w x + \sigma^2       
                    
        p( w | sigma) = N( w | 0, sigma*I)
        '''


        for data_pt in range(self.trainX.shape[0]):
            #Estimate the new posterior
            #print self.trainX[data_pt]
            print 'weights', self.weights
            kernelResult = self.eval_kernel(self.weights, self.supportVecs, self.trainX[data_pt])
            print 'kernelResult' , kernelResult
            print 'label', self.trainY[data_pt]
            err =  1 - (self.trainY[data_pt] * kernelResult)
            print 'err', err
            if err > 0:

                #print self.supportVecs.shape
                #print self.supportVecs
                #print self.trainX[data_pt]
                self.supportVecs = np.concatenate((self.supportVecs.T, np.array([self.trainX[data_pt]]))).T
                #print self.supportVecs
                print 'weights', self.weights
                this = np.array( [0.4 * self.trainY[data_pt] * (1-kernelResult)])
                print 'supportVecs', self.supportVecs
                self.weights = np.concatenate((self.weights, this))



        
        
    def eval_kernel(self, weights, supportvecs, dataX):
        s = 0
        print 'supportvecs shape', supportvecs.shape
        for x in range(supportvecs.shape[1]):
            #for y in range(dataX.shape[0]):
                # rbf
                #print x
                #print y
                #print weights[x]
            diff = dataX - supportvecs[:,x]
            s += weights[x] * np.exp(-np.linalg.norm(diff) / self.sigma)

        return s
    
    def predict(self, dataX):
        #Check if sane data point
        assert(dataX.shape == self.trainX[0].shape)
        #@todo: Integrate over weights distribution for this data point, or just take the mean?
        #Sample from the likelihood normal
        return np.random.normal(np.dot(self.mean, dataX), self.observation_error_sigma) 
        
    def test(self, dataX, true_label):
        if self.predict(dataX)*true_label > 0:
            return True
        else:
            return False
    
    def linear_feature(self, dataX):
        return np.append(dataX, 1.0)


if __name__ == "__main__":
    #Test the regression
    
    #generate synthetic data from  f = -0.3 + 0.5x with 0.2 sigma noise
    num = 30
    Xs = np.append(np.vstack(np.random.uniform(size=num)), np.ones((num,1)), 1)
    Ys = np.dot(Xs, [-0.3, 0.5]) + np.random.normal(0, 0.2, size=num)
    print Ys
    Ys[Ys<0.4] = -1
    Ys[Ys>0.4] = 1

    bl_params = [0.1, 0, 1.0]
    test_regression = OKSVM(Xs, Ys, bl_params)
    test_regression.train()

    #print test_regression.predict(np.array([0.5, 1])), test_regression.test(np.array([0.5, 1]), 1)