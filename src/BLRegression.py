#!/usr/bin/env python 
'''
An implementation of Bayesian Linear Regression/Classification
'''

import numpy as np
from scipy.linalg.basic import inv

class BLRegression:
    
    def __init__(self, features, classes, params):
        assert(len(params) == 3)
        self.trainX = features
        self.trainY = classes
        
        self.observation_error_sigma = params[0]
        self.prior_mean = params[1]*np.ones(self.trainX.shape[1]);
        self.prior_sigma = params[2];
        
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
        #Initialize the weights
        self.weights = np.zeros((self.trainX.shape[1], 1))
        #Initialize the Gaussian prior on y
        weight_prior_P = 1.0/(self.prior_sigma**2) * np.diag(np.ones(self.trainX.shape[1]))
        weight_prior_J = np.dot(weight_prior_P, self.prior_mean)
        #In a loop, for every training point, update the posterior
        for data_pt in range(self.trainX.shape[0]):
            #Estimate the new posterior
            (weight_prior_P, weight_prior_J) = self.eval_posterior(weight_prior_P, weight_prior_J, self.observation_error_sigma, self.trainX[data_pt], self.trainY[data_pt])        
        #Store the final parameters for the weight vector ditribution
        self.P = weight_prior_P
        self.J = weight_prior_J
        self.cov = inv(self.P)
        self.mean = np.dot(self.cov, self.J)
        
        
    def eval_posterior(self, weight_prior_P, weight_prior_J, observation_error_sigma,  dataX, dataY):
        J_new = np.dot(dataX, dataY)/observation_error_sigma**2 + weight_prior_J
        P_new = np.outer(dataX, dataX)/observation_error_sigma**2 + weight_prior_P
        return (P_new, J_new)
    
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
    num = 1000
    Xs = np.append(np.vstack(np.random.uniform(size=num)), np.ones((num,1)), 1)
    Ys = np.dot(Xs, [-0.3, 0.5]) + np.random.normal(0, 0.2, size=num)
    
    bl_params = [0.1, 0, 1.0]
    test_regression = BLRegression(Xs, Ys, bl_params)
    test_regression.train()
    print("\nTest::Mean Vector = \n"+repr(test_regression.mean)+"Test::Covariance = \n"+repr(test_regression.cov))
    
    print test_regression.predict(np.array([0.5, 1])), test_regression.test(np.array([0.5, 1]), 1)