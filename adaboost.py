#!/usr/bin/python

import sys
import numpy as np
import math
import datetime


class Adaboost:
    def __init__(self, train_data):
        self.id = []
        self.train = []
        self.labs = []
        self.nrec = len(train_data)
        self.weights = np.array([[1 / float(self.nrec)] * 4] * self.nrec) # equal weights initialised 1/N to 4 orientation
        self.curr_labs = []
        self.alpha = dict()
        self.learners = dict()
        self.outputs = dict()
        self.iterations = 2 #Increase and check
        self.prediction = []

        #Binary classifier        
        # 0 would be [1,-1,-1,-1] and 90 would be [-1,1,-1,-1]
        for i in train_data:
            self.id.append(i.id)
            self.train.append(np.array(i.features))
            self.labs.append(i.orientation)
            self.curr_labs.append([1 if i.orientation == 0 else -1] + [1 if i.orientation == 90 else -1] + [
                1 if i.orientation == 180 else -1] + [1 if i.orientation == 270 else -1])
        self.curr_labs = np.array(self.curr_labs)

    def train_learners(self):
        '''This function trains the weak learners where a learner outputs 1 if a condition is matched and -1 when it is not.
        Here, the condition is if a pixel is greater than another. Only top two rows of pixels are compared with the bottom two to improve performance.'''
        print "started training learners at", datetime.datetime.now()
        for i in range(20):
            for j in range(172, 192):
                cname = (i, j)  # classifier i>j
                output = np.array([1 if x[i] > x[j] else -1 for x in self.train])
                self.outputs[cname] = output
       
        print "finished training learners at", datetime.datetime.now()

    def get_best_learner(self):
        '''This function finds the weak classifiers for each of 0,90,180 and 270 binary classifiers basis the no. of examples that are misclassified.
        Error rate is calculated for each of the classifiers and the one with minimum error is chosen.'''
        print "finding best learners:", datetime.datetime.now()
        err_0 = dict()
        err_90 = dict()
        err_180 = dict()
        err_270 = dict()

        for learner in self.outputs:
            #x is product of trained data and current orientation y is weight , if error is smaller the higher will the weight. For misclassification weights should be higher
            err_0[learner] = sum([y if x == -1 else 0 for x, y in zip(self.outputs[learner] * self.curr_labs[:, 0], self.weights[:, 0])]) / float(self.nrec)
            err_90[learner] = sum([y if x == -1 else 0 for x, y in zip(self.outputs[learner] * self.curr_labs[:, 1], self.weights[:, 1])]) / float(self.nrec)
            err_180[learner] = sum([y if x == -1 else 0 for x, y in zip(self.outputs[learner] * self.curr_labs[:, 2], self.weights[:, 2])]) / float(self.nrec)
            err_270[learner] = sum([y if x == -1 else 0 for x, y in zip(self.outputs[learner] * self.curr_labs[:, 3], self.weights[:, 3])]) / float(self.nrec)

        value_0 = list(err_0.values())
        key_0 = list(err_0.keys())

        value_90 = list(err_90.values())
        key_90 = list(err_90.keys())

        value_180 = list(err_180.values())
        key_180 = list(err_180.keys())

        value_270 = list(err_270.values())
        key_270 = list(err_270.keys())

        best_0 = key_0[value_0.index(min(value_0))]
        best_90 = key_90[value_90.index(min(value_90))]
        best_180 = key_180[value_180.index(min(value_180))]
        best_270 = key_270[value_270.index(min(value_270))]

        # Alpha, i.e. a weights to be given to each weak classifier is calculated basis the error rate
        # and is used to predict the final output.
        print "calculating alphas at", datetime.datetime.now()
        #log(1-error/error)
        a_0 = 0.5 * math.log((1 - min(value_0)) / min(value_0))
        a_90 = 0.5 * math.log((1 - min(value_90)) / min(value_90))
        a_180 = 0.5 * math.log((1 - min(value_180)) / min(value_180))
        a_270 = 0.5 * math.log((1 - min(value_270)) / min(value_270))
        return [(best_0, a_0), (best_90, a_90), (best_180, a_180), (best_270, a_270)]

    def update_weights(self, learners):
        '''This function updates the weights of samples basis the classification.
        The misclassified examples are given more weight and the correctly classified ones are given less weight.'''
        print "updating weights at", datetime.datetime.now()
        l_0 = learners[0][0]
        a_0 = learners[0][1]
        l_90 = learners[1][0]
        a_90 = learners[1][1]
        l_180 = learners[2][0]
        a_180 = learners[2][1]
        l_270 = learners[3][0]
        a_270 = learners[3][1]

        wt_0 = [self.weights[:, 0] * np.exp(-a_0 * self.outputs[l_0] * self.curr_labs[:, 0])]
        self.weights[:, 0] = wt_0 / np.sum(wt_0)

        wt_90 = [self.weights[:, 1] * np.exp(-a_90 * self.outputs[l_90] * self.curr_labs[:, 1])]
        self.weights[:, 1] = wt_90 / np.sum(wt_90)

        wt_180 = [self.weights[:, 2] * np.exp(-a_180 * self.outputs[l_180] * self.curr_labs[:, 2])]
        self.weights[:, 2] = wt_180 / np.sum(wt_180)

        wt_270 = [self.weights[:, 3] * np.exp(-a_270 * self.outputs[l_270] * self.curr_labs[:, 3])]
        self.weights[:, 3] = wt_270 / np.sum(wt_270)
        print "finished updating weights at", datetime.datetime.now()

    def predict(self, cl_list):
        ''' This function predicts the final classification basis alpha values of each of the classifiers and sums the total value.
        '''
        output = np.array([[float(0)] * 4] * self.nrec)
        for classify in cl_list:
            classify_0 = classify[0][0]
            a_0 = classify[0][1]

            classify_90 = classify[1][0]
            a_90 = classify[1][1]

            classify_180 = classify[2][0]
            a_180 = classify[2][1]

            classify_270 = classify[3][0]
            a_270 = classify[3][1]

            output[:, 0] += self.outputs[classify_0] * a_0
            output[:, 1] += self.outputs[classify_90] * a_90
            output[:, 2] += self.outputs[classify_180] * a_180
            output[:, 3] += self.outputs[classify_270] * a_270
        return output

    def predict_orientation(self, output):
        '''This function predicts the orientation basis the weight each example has for each classfier output. If classifier for 0 has highest weight
        among classifiers for 90, 180 and 270, it would be classified as 0. '''
        orientation = []
        for test in output:
            ort = test.tolist()
            orientation.append(ort.index(max(ort)))
        orientation = [0 if ort == 0 else 90 if ort == 1 else 180 if ort == 2 else 270 for ort in orientation]
        return orientation

    def check_accuracy(self, orientation):
        cnt = 0
        for i, j in zip(self.labs, orientation):
            if i == j:
                cnt += 1
        print "Accuracy=", cnt / float(self.nrec)

    def adaboost(self):
        '''This function trains the classifiers and chooses best classifiers iteratively.'''
        print datetime.datetime.now()
        self.train_learners()
        classifier_list = []
        for t in range(self.iterations):
            print "\nIteration", t, "started at", datetime.datetime.now()
            l = self.get_best_learner()
            classifier_list.append(l)
            self.update_weights(l)
        print "finished training at", datetime.datetime.now()
        return classifier_list

    def adaboost_test(self, train_data, params):
        self.train_learners()
        output = self.predict(params)
        orientation = self.predict_orientation(output)
        for i, j in zip(train_data, orientation):
            i.pred_orientation = j