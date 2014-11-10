#!/usr/bin/env python
'''
An implementation of Multiclass Winnow. Uses a set of weights for each of the classes. Accepts only binary features.
'''

import numpy as np
from scipy.linalg.basic import inv
from LogReader import LogReader

class MCWinnow:

    def __init__(self, features, classes):
        # Assert that features are binary
        assert(np.array_equal(np.unique(features),np.array([0,1])))
        self.trainX = features
        self.trainY = classes
        self.class_names = np.unique(classes)
        self.num_features = self.trainX.shape[1]
        self.num_classes = len(self.class_names)


        # those could be used as parameters
        self.thresh = self.num_features
        self.p = 2
        self.d = 0.5

    def print_info(self):

        print "[MulticlassWinnow] Number of observations: ", self.trainY.shape[0]
        print "[MulticlassWinnow] Number of features: ", self.num_features
        print "[MulticlassWinnow] Number of classes: ", self.num_classes
        print "[MulticlassWinnow] Promotion parameter: ",self.p
        print "[MulticlassWinnow] Demotion parameter: ",self.d
        # add function to print info



    def train(self):

        #Initialize all the weights to 1
        self.weights = np.ones([self.num_features,self.num_classes])
        # Track number of mistakes
        err_num = 0

        for index in range(self.trainX.shape[0]):
            (predicted_label,scores) = self.predict(self.trainX[index,:])
            self.update_weights(scores, index)
            if not(predicted_label == self.trainY[index]):
                err_num += 1
        print "[MulticlassWinnow] Number of mistakes at learning: ", err_num


    def predict(self,data_pt):
        score = np.zeros([self.num_classes,1])
        # for every class
        for i in range(self.num_classes):
            # compute dot product
            score[i] = np.dot(data_pt, self.weights[:,i])

        return np.argmax(score), score

    def update_weights(self,score,index):
        # for the wrongly labeled class, find the features ==1, and demote or promote their weights depending on type
        # of misclassification

        true_label = self.trainY[index]
        which_indices = (self.trainX[index,:]==1)
        for i in range(self.num_classes):
            if (score[i]>self.thresh and not(true_label==i)):
                self.weights[which_indices,i] = self.weights[which_indices,i]*self.d
            elif (score[i]<=self.thresh and (true_label==i)):
                self.weights[which_indices,i] = self.weights[which_indices,i]*self.p


    def test(self, data_pt,true_label):

        predicted_label = self.predict(data_pt)
        if predicted_label == true_label:
            return True
        else:
            return False

def convert_to_binary(features,classes):
    (n,d) = features.shape
    binary_features = np.array(features)
    feat_threshold = np.zeros(d)
    for j in range(d):
        sorted_tmp = np.sort(features[:,j])
        original_index = np.argsort(features[:,j])
        cut_entropy = np.zeros([n-1,1])
        cut_index = []
        counter = 0
        print "computing binary features for class", j
        for i in range(n-1):
            # only look at boundary points
            if not(classes[original_index[i]]==classes[original_index[i+1]]):
                # choose cutpoint as the midpoint between consecutive points
                T = (sorted_tmp[i]+sorted_tmp[i+1])/2.
                index_1 = np.where(binary_features[:,j]<T)
                index_2 = np.where(binary_features[:,j]>T)
                ent1 = compute_entropy(classes[index_1])
                ent2 = compute_entropy(classes[index_2])
                cut_entropy[counter] = (len(index_1)/n)*ent1 + (len(index_2)/n)*ent2
                cut_index.append(T)
                counter += 1
        print "Number of boundary points for class ",j,": ",counter-1
        best_threshold = np.argmin(cut_entropy[0:counter-1])
        feat_threshold[j] = cut_index[best_threshold]
        # threshold based on best cutpoint
        binary_features[binary_features[:,j]<feat_threshold[j],j] = 0
        binary_features[binary_features[:,j]>feat_threshold[j],j] = 1
    return binary_features,feat_threshold

def compute_entropy(classes):
    class_id = np.unique(classes)
    n = float(len(classes))
    p = np.zeros(class_id.size)
    for i in range(len(class_id)):

        p[i] = len(np.where(classes == class_id[i]))/ n

    return -np.dot(p,np.log2(p))

def threshold_to_binary(features,feat_threshold):
    (n,d) = features.shape

    for j in range(d):
        binary_features[binary_features[:,j]<feat_threshold[j],j] = 0
        binary_features[binary_features[:,j]>feat_threshold[j],j] = 1
    return binary_features

if __name__ == "__main__":

    train_log_object = LogReader('../data/oakland_part3_am_rf.node_features')
    train_points = train_log_object.read()

    Xs = np.array([point._feature for point in train_points])
    Ys = np.array([point._label for point in train_points])
    binary_features = np.load('am_binary_features.npy')
    feat_threshold =  np.load('am_binary_threshold.npy')
    #(binary_features,feat_threshold) = convert_to_binary(Xs,Ys)
    #np.save("am_binary_features",binary_features)
    #np.save("am_binary_threshold",feat_threshold)

    test_regression = MCWinnow(binary_features,Ys)
    test_regression.train()
    #print "[MultiClassWinnow] Learned weights \n", test_regression.weights
    test_log_object = LogReader('../data/oakland_part3_an_rf.node_features')
    test_points = test_log_object.read()
    testXs = np.array([point._feature for point in test_points])
    testYs = np.array([point._label for point in test_points])
    n = testYs.shape[0]
    binary_test_features = threshold_to_binary(testXs,feat_threshold)
    test_err = 0
    for index in range(n):
        if test_regression.test(binary_test_features[index,:],testYs[index]):
            test_err += 1
    print "Number of mistakes at testing ",test_err
