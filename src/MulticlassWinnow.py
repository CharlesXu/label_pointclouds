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



        self.thresh = self.num_features
        # those could be used as parameters
        self.p = 2
        self.d = 0.5

    def print_info(self):

        print "[MulticlassWinnow] Number of observations: ", self.trainY.shape[0]
        print "[MulticlassWinnow] Number of features: ", self.num_features
        print "[MulticlassWinnow] Number of classes: ", self.num_classes
        print "[MulticlassWinnow] Promotion parameter: ", self.p
        print "[MulticlassWinnow] Demotion parameter: ", self.d
        # add function to print info

    def train(self):

        #Initialize all the weights to 1
        self.weights = np.ones([self.num_features, self.num_classes])
        # Track number of mistakes
        err_num = np.zeros([self.num_classes,1])

        for index in range(self.trainX.shape[0]):
            (predicted_label,scores) = self.predict(self.trainX[index,:])
            if not(predicted_label == self.trainY[index]):
                err_num[self.trainY[index]] += 1
                #print "predicted", predicted_label
                self.update_weights(scores, index)

        print "[MulticlassWinnow] Number of mistakes at learning: ", err_num

    def predict(self, data_pt):
        score = np.zeros([self.num_classes, 1])
        # for every class
        for i in range(self.num_classes):
            # compute dot product
            score[i] = np.dot(data_pt, self.weights[:,i])

        return np.argmax(score), score

    def update_weights(self,score,index):
        # for the wrongly labeled class, find the features ==1, and demote or promote their weights depending on type
        # of misclassification
        true_label = self.trainY[index]
        which_indices = np.where(self.trainX[index,:] == 1)

        #self.weights[which_indices,predicted_label] *= self.d

        #self.weights[which_indices,true_label] *= self.p
        for i in range(self.num_classes):
            if score[i]>= self.thresh and not(true_label==i):
                #print index," down ",i,true_label,score[i],self.trainX[index,:]*self.weights[:,i]
                self.weights[which_indices,i] *= self.d
            elif score[i]< self.thresh and (true_label==i):
                #print index," up ",i,true_label,score[i],self.trainX[index,:]*self.weights[:,i]
                self.weights[which_indices,i] *= self.p

    def test(self, data_pt,true_label):

        (predicted_label,tmp) = self.predict(data_pt)
        if predicted_label == true_label:
            return True
        else:
            return False


def convert_to_binary(features,classes):
    (n,d) = features.shape
    binary_features = np.array(features)
    feat_threshold = np.zeros(d)

    for j in range(d):
        potential_cutoffs = np.unique(features[:,j])
        inner_iter_loop = len(potential_cutoffs)
        print "Number of unique values for feature ",j," : ", inner_iter_loop
        if len(potential_cutoffs)>2:
            sorted_tmp = np.sort(features[:,j])
            original_index = np.argsort(features[:,j])
            cut_entropy = np.zeros([inner_iter_loop-1,1])
            cut_index = []
            counter = 0
            #print "computing binary features for feature", j
            for i in range(inner_iter_loop-1):
                # only look at boundary points
                classpool = np.hstack((classes[np.where(features[:,j]== potential_cutoffs[i])], classes[np.where(features[:,j]==potential_cutoffs[i+1])]))

                if not(len(np.unique(classpool))==1):
                    # choose cutpoint as the midpoint between consecutive points
                    T = (potential_cutoffs[i]+potential_cutoffs[i+1])/2.
                    index_1 = np.where(binary_features[:,j]<=T)
                    index_2 = np.where(binary_features[:,j]>T)
                    ent1 = compute_entropy(classes[index_1])
                    ent2 = compute_entropy(classes[index_2])
                    cut_entropy[counter] = (len(index_1)/float(n))*ent1 + (len(index_2)/float(n))*ent2
                    cut_index.append(T)
                    counter += 1
            print "Number of boundary points for feature ",j,": ",counter-1
            best_threshold = np.argmin(cut_entropy[0:counter-1])
            print "min and max entropy for feature ",j,": ",np.min(cut_entropy[0:counter-1]), np.max(cut_entropy[0:counter-1])
            feat_threshold[j] = cut_index[best_threshold]
            print "Chosen cutoff: ", feat_threshold[j]
            binary_features[np.where(binary_features[:,j]<= feat_threshold[j]),j] = int(0)
            binary_features[np.where(binary_features[:,j]>feat_threshold[j]),j] = int(1)
        elif len(potential_cutoffs) == 2:
            print "Feature ", j, " is already binary"
            feat_threshold[j] = np.mean(potential_cutoffs)
            # threshold based on best cutpoint
            binary_features[np.where(binary_features[:,j]<= feat_threshold[j]),j] = int(0)
            binary_features[np.where(binary_features[:,j]>feat_threshold[j]),j] = int(1)
        else:
            feat_threshold[j] = float('nan')
            binary_features[:, j] = 0

    return binary_features,feat_threshold

def compute_entropy(classes):
    class_id = np.unique(classes)
    n = float(len(classes))
    p = np.zeros(class_id.size)
    for i in range(len(class_id)):
        tmp = np.where(classes == class_id[i])
        p[i] = len(tmp[0])/n

    return -np.dot(p,np.log2(p))

def threshold_to_binary(features,feat_threshold):
    (n,d) = features.shape
    binary_features = features
    for j in range(d):
        binary_features[binary_features[:,j]<= feat_threshold[j],j] = 0
        binary_features[binary_features[:,j]> feat_threshold[j],j] = 1
    return binary_features

if __name__ == "__main__":

    train_log_object = LogReader('../data/oakland_part3_an_rf.node_features')
    binary_features = np.load('an_binary_features2.npy')
    feat_threshold =  np.load('an_binary_threshold2.npy')
    test_log_object = LogReader('../data/oakland_part3_am_rf.node_features')

    train_points = train_log_object.read()

    Xs = np.array([point._feature for point in train_points])
    Ys = np.array([point._label for point in train_points])
    #(binary_features,feat_threshold) = convert_to_binary(Xs,Ys)
    #np.save('am_binary_features2', binary_features)
    #np.save('am_binary_threshold2', feat_threshold)
    #binary_features = threshold_to_binary(Xs,feat_threshold)
    #train_index = np.random.randint(0,Xs.shape[0],10)
    train_index =   np.random.permutation(Xs.shape[0]) #range(Xs.shape[0]) #
    test_regression = MCWinnow(binary_features[train_index, :], Ys[train_index])
    test_regression.train()
    print "[MultiClassWinnow] Learned weights \n", test_regression.weights
    test_points = test_log_object.read()
    testXs = np.array([point._feature for point in test_points])
    testYs = np.array([point._label for point in test_points])

    n = testYs.shape[0]
    binary_test_features = threshold_to_binary(testXs,feat_threshold)
    confusion_matrix = np.zeros([test_regression.num_classes, test_regression.num_classes],dtype=int)
    test_err =0
    for index in range(n):
        (predicted_label, tmp) = test_regression.predict(binary_test_features[index,:])
        if not(predicted_label == testYs[index]):
            #print predicted_label,testYs[index]
            confusion_matrix[testYs[index],predicted_label] += 1
            test_err +=1
        else:
            confusion_matrix[testYs[index],testYs[index]] +=1
    print "confusion matrix \n", confusion_matrix
    print "percent accuracy", 100*(1-test_err/float(n))
