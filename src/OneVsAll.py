#!/usr/bin/env python 
'''
A container class for running OneVsAll operations for a generic binary classifier that implements
train()
predict(feature_vector) -> confidence_measure (/regressed value)
test(feature_vector, true_label) -> boolean

see BLRegression.py for an example
'''
import numpy as np
from Point import Point
from LogReader import LogReader
from BLRegression import BLRegression
from binaryWinnow import binaryWinnow
from binaryWinnow import threshold_to_binary
from binaryWinnowvar import binaryWinnowvar
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pdb
from OnlineKernelSVM import OKSVM
import plot_points

class OneVsAll:
    def __init__(self, features, labels, classifier_class, classifier_params, test_features = None, test_labels = None):
               
        if test_features is None:
            self.X = np.array(features)
            self.Y = np.array(labels)
            
            hold_out_index = self.X.shape[0]*0.8
            
            self.trainX = self.X[:hold_out_index,:]
            self.trainY = self.Y[:hold_out_index]
            
            self.testX  = self.X[hold_out_index:, :]
            self.testY  = self.Y[hold_out_index:]
        else:
            self.trainX = np.array(features)
            self.trainY = np.array(labels)
            
            self.testX = np.array(test_features)
            self.testY = np.array(test_labels)
        
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params
        self.classifiers = []
        
        
    def train(self):
        print '[OneVsAll] Training individual predictors...'
        for label in range(len(Point.label_dict)):
#         for label in [0,3]:
            #Separate the data into positive and negative classes
            trainY = self.trainY.copy()
            trainX = self.trainX.copy()
            if not(self.classifier_class == binaryWinnow):
                trainY[trainY != label] = -1
                trainY[trainY != -1] = 1
                trainY[trainY == -1] = 0
            else:
                trainY[trainY != label] = 10
                trainY[trainY != 10] = 1
                trainY[trainY != label] = 0
            #Rebalance the data
            (trainX,trainY) = correct_imbalance(trainX,trainY)
            trainY[trainY == 0] = -1
            trainY[trainY == 1] = 1
            #print 'shapes::',trainY[trainY==-1].shape, trainY[trainY==1].shape
            trainY = trainY[:,0]
            #print trainY.shape, trainX.shape
            #Train the classifier
            classifier = self.classifier_class(trainX, trainY, self.classifier_params)
            classifier.train()
            self.classifiers.append([classifier, label])
            print '[OneVsAll] Trained ', Point.label_rev_dict[label]
        print '[OneVsAll] Done!'
        
    def predict(self, dataX):
        #Check if sane data point
        assert(dataX.shape == self.trainX[0].shape)
        predicted_confidences = [classifier[0].predict(dataX) for classifier in self.classifiers]
        best_classifier = np.argmax(np.abs(predicted_confidences))
        return self.classifiers[best_classifier][1]
    
    def test(self):
        evals = []
        true_labels = []
        predicted_labels = []
        for index in range(len(self.testX)):
            dataX = self.testX[index]
            true_label = self.testY[index]
#             if true_label not in [0,3]:
#                 continue
            predicted_label = self.predict(dataX)
#             evals.append(self.classifiers[predicted_label].test(dataX, true_label))
            evals.append(predicted_label == true_label) 
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        #Now evaluate accuracy
        #True == 1, thus sum
        print '[OneVsAll] Accuracy = ', float(sum(evals))/len(evals)
        #Generate confusion matrix
        labels = [Point.label_rev_dict[i] for i in range(len(Point.label_dict))]
#         labels = [Point.label_rev_dict[i] for i in [0,3]]
        
        cm =  confusion_matrix(true_labels, predicted_labels )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for x in xrange(len(cm)):
            for y in xrange(len(cm)):
                ax.annotate(str(cm[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
        plt.show()
        
        return predicted_labels

    def cvtest(self,testX,testY):
        evals = []
        true_labels = []
        predicted_labels = []
        for index in range(len(testX)):
            dataX = testX[index]
            true_label = testY[index]
#             if true_label not in [0,3]:
#                 continue
            predicted_label = self.predict(dataX)
#             evals.append(self.classifiers[predicted_label].test(dataX, true_label))
            evals.append(predicted_label == true_label)
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        #Now evaluate accuracy
        #True == 1, thus sum
        print '[OneVsAll] Accuracy = ', float(sum(evals))/len(evals)
        acc = float(sum(evals))/len(evals)
        #Generate confusion matrix
        labels = [Point.label_rev_dict[i] for i in range(len(Point.label_dict))]
#         labels = [Point.label_rev_dict[i] for i in [0,3]]

        cm =  confusion_matrix(true_labels, predicted_labels )
        #print "confusion matrix, ",cm
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #cax = ax.matshow(cm)
        #plt.title('Confusion matrix of the classifier')
        #fig.colorbar(cax)
        #ax.set_xticklabels([''] + labels)
        #ax.set_yticklabels([''] + labels)
        #plt.xlabel('Predicted')
        #plt.ylabel('True')
        #for x in xrange(len(cm)):
        #    for y in xrange(len(cm)):
        #        ax.annotate(str(cm[x][y]), xy=(y, x),
        #                    horizontalalignment='center',
        #                    verticalalignment='center')
        #plt.show()

        return predicted_labels,cm,acc

def correct_imbalance(Xs,Ys):
    class_id = np.unique(Ys)
    h = np.bincount(Ys)
    d = np.std(h[0])
    mode = np.max(h)
    #print "most numerous class is ", np.argmax(h), " it has ", mode, " points"
    newXs = Xs.copy()
    newYs = Ys.copy()
    newYs.shape = (Xs.shape[0],1)
    golden_ratio = 0.2
    for i in range(len(class_id)):
        ratio_to_mode = h[i]/float(mode)
        if ratio_to_mode <golden_ratio:
            #print "adding ",int((golden_ratio - ratio_to_mode)*mode)," points to class",i
            index = np.where(Ys==i)
            tmp_index = np.random.random_integers(0,len(index),int((golden_ratio - ratio_to_mode)*mode))
            newXs = np.vstack((newXs,Xs[index[0][tmp_index],:]))

            tmp = i*np.ones([int((golden_ratio - ratio_to_mode)*mode),1])
            newYs = np.vstack((newYs, tmp))
    #And shuffle!
    shuffled_indices = range(len(newYs))
    np.random.shuffle(shuffled_indices)
    newXs = newXs[shuffled_indices, :]
    newYs = newYs[shuffled_indices, :]
    return (newXs,newYs)



if __name__ == "__main__":
    #Sample implementation for a Bayes Linear Classifier
    #Load a log
    train_log_object = LogReader('../data/oakland_part3_an_rf.node_features')
    train_points = train_log_object.read()
    train_binary_features = np.load('an_binary_features2.npy')
    feat_threshold =  np.load('an_binary_threshold2.npy')
    test_log_object = LogReader('../data/oakland_part3_am_rf.node_features')
    test_points = test_log_object.read()
    test_binary_features = threshold_to_binary(np.array([point._feature for point in test_points]),feat_threshold)

    bl_params = [0.2, 0.0, 1.0]

    trainXs = np.array([point._feature for point in train_points])
    trainYs = np.array([point._label for point in train_points])
    testXs = np.array([point._feature for point in test_points])
    testYs = np.array([point._label for point in test_points])


#     (X,Y) = correct_imbalance(trainXs,trainYs)
    

#     orchestrator = OneVsAll([point._feature for point in train_points], [point._label for point in train_points], BLRegression)

#     orchestrator = OneVsAll(train_binary_features, [point._label for point in train_points],
#                             binaryWinnow, bl_params,
#                             test_binary_features, [point._label for point in test_points])
#     orchestrator = OneVsAll([point._feature for point in train_points], [point._label for point in train_points],
#                             BLRegression, bl_params,
#                             [point._feature for point in test_points], [point._label for point in test_points])
    #orchestrator = OneVsAll(train_binary_features, [point._label for point in train_points],
    #                        binaryWinnow, bl_params,
    #                        test_binary_features, [point._label for point in test_points])
    orchestrator = OneVsAll(trainXs, trainYs,binaryWinnowvar, [10,0.01],testXs, testYs)
    #orchestrator = OneVsAll(trainXs, trainYs,BLRegression, bl_params,testXs, testYs)
    #orchestrator = OneVsAll(train_binary_features, [point._label for point in train_points],
    #                         binaryWinnow, [10,0.01],
    #                         test_binary_features, [point._label for point in test_points])
#     orchestrator.train()
#     orchestrator.test()


    #print 'And now Linear Kernel SVM'
    #svm_params = [0.004, 'linear', 0.01]
    #orchestrator = OneVsAll(trainXs, trainYs,
    #                         OKSVM, svm_params,
    #                         testXs, testYs)

    orchestrator.train()
    predicted_labels = orchestrator.test()
    plot_points.plot_predicted_labels(test_points, predicted_labels)
