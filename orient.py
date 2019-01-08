#!/usr/bin/env python
# Problem Statement : Machine Learning
# a2.py
# Author : rojraman-sgodshal-araghuv
# 
# Implemented a Knn, Adaboost and Random forest algorithms for image classifications 
#References: https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249

import sys
import nearest
from image import Image
import cPickle as Pickle
from time import time
from adaboost import Adaboost
import operator
import random
import pandas as pd
from collections import Counter
import math
import numpy as np


#Random Forest Implementation
# Node class for creating nodes in tree
class Node: 
    def __init__(self,key): 
        self.left = None
        self.right = None
        self.check = key 
        self.decision = None


#reading training and testing data and creating dataframe for the same
def readData(path):
    col_list = ["value"]
    for i in range(64):
        col_list.append("r"+str(i))
        col_list.append("g"+str(i))
        col_list.append("b"+str(i))
    f=open(path,'r')
    train_data = []
    image_name_list = []
    for line in f:
        line=line.strip()
        image_name_list.append(line.split()[0])
        checkval_list = [line.split()[1]]
        for val in line.split()[2:]:
            checkval_list.append("low") if int(val) <= 128 else checkval_list.append("high")
        train_data.append(checkval_list)
    train_dataset = pd.DataFrame(train_data,columns=col_list)
    return train_dataset,image_name_list


#making decision trees 
def make_decision_tree(data,predicate):
    a = data.groupby('value').size().to_dict()
    #return node if pure set
    if len(a) == 1 :
        temp = Node("decision")
        temp.decision=a.keys()[0]
        return temp
    #return node if leaf node not reached and return value with maximum freq
    if len(predicate)==0:
        temp = Node("decision")
        temp.decision=max(a.iteritems(), key=operator.itemgetter(1))[0]
        return temp
    #select predicate with maximum infogain
    selected_predicate = find_predicate(data,predicate)
    # create selected predicate root
    root = Node(selected_predicate)
    # remove selected predicate from list
    predicate.remove(selected_predicate)
    split_data={}
    for region, df_region in data.groupby(selected_predicate):
        df_region.drop(selected_predicate, axis=1, inplace=True)
        split_data[region] = df_region
    # recurively making child of trees
    if 'low' in split_data:
        root.left = make_decision_tree(split_data['low'],predicate)
    if 'high' in split_data:
        root.right = make_decision_tree(split_data['high'],predicate)
    return root

#calculating data entropy -> sumof P(x)*log(P(x))
def data_entropy(data):
    a = data.groupby('value').size().to_dict()
    count = 0
    for col in a.keys():
        prob=a[col]/float(sum(a.values()))
        count+=prob*math.log(prob)
    return -count

#calculating data entropy -> sumof P(x)*log(P(x)) for each predicate
def entropy(data,att):
    a = data.groupby(att).size().to_dict()
    count = 0
    for region, df_region in data.groupby(att):
        prob = a[region]/float(sum(a.values()))
        count+=(a[region]/float(sum(a.values()))) * data_entropy(df_region)
    return count

#calculating info gain for predicates and returning predicate with maximum value
def find_predicate(data,predicate):
    dataset_entropy = data_entropy(data)
    max_count = -999
    selected_predicate = ''
    for att in predicate:
        entropy_att = entropy(data,att)
        sub = dataset_entropy - entropy_att
        if sub > max_count:
            max_count = sub
            selected_predicate = att
    return selected_predicate

#traverse tree function for finding answer in the tree
def traverse_tree(root,test):
    if root == None :
        return -1
    else:
        if root.check =='decision':
            return root.decision
        else:
            val = test[root.check]
            if val == 'low':
                return traverse_tree(root.left,test)
            else:
                return traverse_tree(root.right,test)

#training data to creating data sample with 10000 rows and 30 predicates and making 30 decision trees
def train_data(train):
    labels = [0,90,180,270]
    forest=[]
    for x in range(30):  
        predicate =list(train.columns.values)
        random.shuffle(predicate)
        sample_dataset=train.sample(n=10000)[predicate[:30]]
        predicate =list(sample_dataset.columns.values)
        if 'value' not in sample_dataset.columns:
            sample_dataset['value']=train['value']
        tree = make_decision_tree(sample_dataset,predicate)
        forest.append(tree)
    return forest

#testing data and comparing with original values
def test_forest(forest,test_data,image_name_list):
    count = 0
    total = len(test1.index)
    output_str=''
    for index,row in test1.iterrows():
        test_value = row['value']
        freq={}
        for elem in forest:
            answer = traverse_tree(elem,row.to_dict())
            if answer != -1:
                if answer in freq:
                    freq[answer] +=1
                else:
                    freq[answer] = 1
        calculate_value=0
        if len(freq)>0:
            calculate_value =  max(freq.iteritems(), key=operator.itemgetter(1))[0]
        # prin l2
        if test_value == calculate_value:
           count = count + 1
        output_str += image_name_list[index] + " "  + str(calculate_value) + "\n" 
    print 'Accuracy :'
    count = count * 1.0
    total = total * 1.0
    print (count/total) * 100.0 
    return output_str

# KNN and Adaboost Implementation
#https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
def pickle_file(obj, filename):
    Pickle.dump(obj, open(filename, 'wb'), protocol=Pickle.HIGHEST_PROTOCOL)


def unpickle_file(filename):
    return Pickle.load(open(filename, 'rb'))

def save_output(img_list, outfile_name):
    with open(outfile_name, 'w') as out_file:
        out_str = ''
        for image in img_list:
            out_str += image.id + ' ' + str(image.pred_orientation) + '\n'
        out_file.write(out_str)


def get_accuracy(img_list):
    correct = 0
    for image in img_list:
        if image.orientation == image.pred_orientation:
            correct += 1
    #print (0.0 + correct) / len(img_list)
    return (0.0 + correct) / len(img_list)


# Main program
if __name__ == '__main__':
    phase = sys.argv[1]  # 'train' phase or 'test' phase
    file_name = sys.argv[2]  # Name of file that contains train images in train phase and test images in test phase
    model_file = sys.argv[3]  # Name of file that contains the training model in test phase
    model = sys.argv[4]  # Model that is to be used to train or test.

    # Parsing the input file and creating the image objects
    if model != "forest" and model != "best":
        image_list = list()
        with open(file_name, 'r') as t_file:
            for line in t_file:
                image_list.append(Image(line))
        print len(image_list)
        starttime = time()
        # K-Nearest neighbors
        if model == 'nearest' and phase == 'train':
            model = nearest.train(image_list)
            #print model
            pickle_file(model, model_file)
        elif model == 'nearest' and phase == 'test':
            model = unpickle_file(model_file)
            nearest.test(image_list, model)

        # ADA boost
        elif model == "adaboost" and phase == "train":
            params = Adaboost(image_list).adaboost()
            pickle_file(params, model_file)
        elif model == "adaboost" and phase == "test":
            params = unpickle_file(model_file)
            Adaboost(image_list).adaboost_test(image_list, params)

        endtime = time()
        print 'time taken : ',(endtime - starttime)
        if phase == 'test':
            accuracy = get_accuracy(image_list)
            save_output(image_list, 'output.txt')
            print 'Accuracy ->', accuracy*100 ,"%"

    else:
        starttime = time()
        if (model == 'forest' or model == 'best') and phase == 'train':
            train1,image_name_list = readData(file_name)
            forest = train_data(train1)
            pickle_file(forest,model_file)
        elif (model == 'forest' or model == 'best') and phase == "test": 
            forest = unpickle_file(model_file)
            test1,image_name_list = readData(file_name)
            out_str=test_forest(forest,test1,image_name_list)
            with open('output.txt', 'w') as out_file:
                out_file.write(out_str)

        endtime = time()
        print 'time taken : ',(endtime - starttime)

