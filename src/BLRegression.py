#!/usr/bin/env python 
'''
An implementation of Bayesian Linear Regression/Classification
'''

import numpy as np
from scipy.linalg.basic import inv
import pdb

class BLRegression:
    
    def __init__(self, features, classes):
        self.X = features
        self.Y = classes
        
        hold_out_index = self.X.shape[0]*0.8
        
        self.trainX = self.X[:hold_out_index,:]
        self.trainY = self.Y[:hold_out_index]
        
        self.testX  = self.X[hold_out_index:, :]
        self.testY  = self.Y[hold_out_index:]
        
        
    def regress(self):
        '''
        The MAP estimate of p(w| D) is the mean of the posterior (which is a gaussian distribution)
        For prediction, we need p(y| x, D)
        
        p(y | x, D) = \int p(y|x, D, w) p(w|x, D) dw
                    = \int p(y|x, w) p(w|D) 
                    = \int N(y|w^Tx, a^-1) N(w | \mu, \lambda^-1)         
                    = N( y| \mu_w^T x , x^T \Sigma_w x + \sigma^2       
                    
        p( w | sigma) = N( w | 0, sigma*I)
        '''
        #Initialize the weights
        self.weights = np.zeros((self.X.shape[1], 1))
        #Initialize the Gaussian prior on y
        observation_error_sigma = 0.2;
        prior_mean = 0.0*np.ones(self.trainX.shape[1]);
        prior_sigma = 1.0;
        weight_prior_P = prior_sigma * np.diag(np.ones(self.X.shape[1]))
        weight_prior_J = np.dot(weight_prior_P, prior_mean)
        #In a loop, for every training point, update the posterior
        for data_pt in range(self.trainX.shape[0]):
            #Estimate the new posterior
            (weight_prior_P, weight_prior_J) = self.eval_posterior(weight_prior_P, weight_prior_J, observation_error_sigma, self.trainX[data_pt], self.trainY[data_pt])        
        
        
        self.P = weight_prior_P
        self.J = weight_prior_J
        self.cov = inv(self.P)
        self.mean = np.dot(self.cov, self.J)
        
        
    def eval_posterior(self, weight_prior_P, weight_prior_J, observation_error_sigma,  dataX, dataY):
        J_new = np.dot(dataX, dataY)/observation_error_sigma**2 + weight_prior_J
        P_new = np.dot(dataX, dataX)/observation_error_sigma**2 + weight_prior_P
        return (P_new, J_new)
    
    def predict(self, dataX):
        #Check if sane data point
        assert(dataX.shape == self.trainX[0].shape)
        #@todo: Integrate over weights distribution for this data point, or just take the mean?
        #Sample from the likelihood normal
        return np.random.normal(np.dot(self.mean, dataX), self.observation_error_sigma)
        
    def linear_feature(self, dataX):
        return np.append(dataX, 1.0)
    
if __name__ == "__main__":
    #Test the regression
    
    #generate synthetic data from  f = -0.3 + 0.5x with 0.2 sigma noise
    num = 100
    Xs = np.append(np.vstack(np.random.uniform(size=num)), np.ones((num,1)), 1)
    Ys = np.dot(Xs, [-0.3, 0.5]) + np.random.normal(0, 0.2, size=num)
    
    test_regression = BLRegression(Xs, Ys)
    test_regression.regress()
    print("\nMean Vector = \n"+repr(test_regression.mean))