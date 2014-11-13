import random as random
import numpy as np
from Point import Point
from LogReader import LogReader
from BLRegression import BLRegression
from binaryWinnow import binaryWinnow
from binaryWinnow import threshold_to_binary
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pdb
from OnlineKernelSVM import OKSVM
import plot_points
from OneVsAll import OneVsAll

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
    train_binary_features = np.load('an_binary_features2.npy')
    feat_threshold =  np.load('an_binary_threshold2.npy')
    test_log_object = LogReader('../data/oakland_part3_am_rf.node_features')
    test_points = test_log_object.read()
    test_binary_features = threshold_to_binary(np.array([point._feature for point in test_points]),feat_threshold)

    Xs = np.array([point._feature for point in train_points])
    Ys = np.array([point._label for point in train_points])
    ds_train = Dataset(Xs,Ys)
    cv_fold = 2
    for i in range(cv_fold):
        print "training fold ",i+1," and testing on the others"
        idx_train = cv_idx_train(ds_train.r, cv_fold, i + 1)
        idx_test = cv_idx_test(ds_train.r, cv_fold, i + 1)
        bl_params = [0.2, 0.0, 1.0]
        orchestrator = OneVsAll(Xs[idx_train,:], Ys[idx_train],
                             BLRegression, bl_params,Xs[idx_test,:], Ys[idx_test])
        orchestrator.train()
        orchestrator.test()



