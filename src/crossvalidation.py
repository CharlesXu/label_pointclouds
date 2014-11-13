import random as random
import numpy as np
from Point import Point
from LogReader import LogReader
from BLRegression import BLRegression
from binaryWinnowvar import binaryWinnowvar
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pdb
from OnlineKernelSVM import OKSVM
import plot_points
from OneVsAll import OneVsAll
from OneVsAll import correct_imbalance

class Dataset:
    def __init__(self, X=[], y=[]):
        self.X = np.array(X)  # r x c matrix of r datapoints, each with c input features
        self.y = np.array(y)  # target output (r x 1 vector)
        if self.X .shape[0]>0:
            self.r = X.shape[0]  # number of datapoints in X
            self.c = X.shape[1]  # number of features
        else:
            self.r = 0
            self.c = 0

    def split(self, nbSplit):
        dsSplit1 = Dataset()
        dsSplit2 = Dataset()
        dsSplit1.X = self.X[0:nbSplit, :]
        dsSplit1.y = self.y[0:nbSplit]
        dsSplit1.r = nbSplit
        dsSplit1.c = self.c
        dsSplit2.X = self.X[nbSplit:self.r, :]
        dsSplit2.y = self.y[nbSplit:self.r]
        dsSplit2.r = self.r - nbSplit
        dsSplit2.c = self.c
        return (dsSplit1, dsSplit2)

    def nfold_split(self, nfold):
        nbEach = int(self.r / float(nfold))
        dsList = list()
        idx = 0
        for i in range(nfold - 1):
            ds = Dataset()
            ds.X = self.X[idx:idx + nbEach, :]
            ds.y = self.y[idx:idx + nbEach]
            ds.r = nbEach
            ds.c = self.c
            idx += nbEach
            dsList.append(ds)
        ds = Dataset()
        ds.X = self.X[idx:self.r, :]
        ds.y = self.y[idx:self.r]
        ds.r = nbEach
        ds.c = self.c
        return dsList

    def merge(self, ds):
        self.X = np.append(self.X, ds.X)
        self.y = np.append(self.y, ds.y)
        self.r += ds.r


def cv_idx_train(n_data, n_fold, fold_id):
    idx = range(0, n_data)
    ratio = 1.0 / n_fold
    last_id = int(n_data * fold_id * ratio)
    first_id = int(n_data * (fold_id - 1) * ratio)
    idx = np.delete(idx, range(first_id, last_id))
    return idx


def cv_idx_test(n_data, n_fold, fold_id):
    ratio = 1.0 / n_fold
    last_id = int(n_data * fold_id * ratio)
    first_id = int(n_data * (fold_id - 1) * ratio)
    return np.array(range(first_id, last_id))


def rand_perm(n):
    p = range(0, n)
    for i in range(0, n):
        r = random.randint(i, n - 1)
        x = p[i]
        p[i] = p[r]
        p[r] = x
    return p


if __name__ == "__main__":

    train_log_object = LogReader('../data/oakland_part3_an_rf.node_features')
    train_points = train_log_object.read()
    test_log_object = LogReader('../data/oakland_part3_am_rf.node_features')
    test_points = test_log_object.read()

    trainXs = np.array([point._feature for point in train_points])
    trainYs = np.array([point._label for point in train_points])
    (X,Y) = correct_imbalance(trainXs,trainYs)

    perm_idx = np.random.permutation(X.shape[0]) #range(X.shape[0])
    ds_train = Dataset(X[perm_idx,:],Y[perm_idx])
    cv_fold = 2
    cm = np.zeros([cv_fold,cv_fold,5,5])
    acc = np.zeros([cv_fold,cv_fold])
    params = [[5,0.01],[10,0.01],[15,0.01]]#,[5,0.001],[10,0.001],[15,0.001],[5,0.1],[10,0.1],[15,0.1]]
    cum_cm = np.zeros([cv_fold,5,5])
    mean_confusion_matrix = np.zeros([3,5,5])
    mean_accuracy = np.zeros([3,1])
    for p in range(3):
        param = params[p]
        acc = np.zeros([cv_fold,cv_fold])
        cm = np.zeros([cv_fold,5,5])
        for i in range(cv_fold):
            print "training fold ",i+1," and testing on the others"
            idx_train = cv_idx_train(ds_train.r, cv_fold, i + 1)
            orchestrator = OneVsAll(X[perm_idx[idx_train],:], Y[perm_idx[idx_train]],binaryWinnowvar, param,[], [])
            orchestrator.train()
            for j in range(cv_fold):
                print "j: ",j
                idx_test = cv_idx_test(ds_train.r, cv_fold, j + 1)
                (predicted_labels,cm[j,:,:],acc[i,j]) = orchestrator.cvtest(X[perm_idx[idx_test],:],Y[perm_idx[idx_test]])
            cum_cm[i,:,:] = np.mean(cm,axis=0)
        mean_confusion_matrix = np.mean(cum_cm,axis=0)
        mean_accuracy = np.mean(acc)

        print "confusion for param ",p," \n ", mean_confusion_matrix
        print "accuracy for param ",p," : ",mean_accuracy



