from sklearn import svm
from sklearn import metrics
from sklearn import multiclass
from sklearn import datasets
from sklearn import preprocessing
import math
import json

from resources import cfg
# import cfg
import numpy as np 
import matplotlib.pyplot as plt

cwd = cfg.get_path()
train_data = []
train_target = []

# Get data from file
with open(cwd + 'sentensesData', 'r') as dataset_file:
    for data in dataset_file:
        tmp = []
        data = data.replace('\n', '')
        data = data.replace('\ufeff', '')
        tmp = data.split('\t')

        for i in range(len(tmp)-1):
            tmp[i] = float(tmp[i])
        tmp[len(tmp)-1] = int(tmp[len(tmp)-1])
        train_data.append(tmp[:len(tmp)-1])
        train_target.append(tmp[-1])
tmp_data = []
tmp_data = train_data.copy()

def normalize_data(data):
    new_tmp_data =tmp_data
    new_tmp_data.append(data)
    new_tmp_data = preprocessing.scale(new_tmp_data)
    return new_tmp_data[-1]


train_data = preprocessing.scale(train_data)
# Test data
train_amount = 75
test_data = train_data[train_amount:].copy()
test_target = train_target[train_amount:].copy()
train_data = train_data[:train_amount]
train_target = train_target[:train_amount]

# One versus Rest
# Class 0 vs Class(1, 2)
iteration = 5000
targatForClassNegative = []
targatForClassNegative = [1 if train_target[i] == 0 else 0 for i in range(len(train_target))]
svmClassNegative = svm.SVC(kernel='linear', gamma='scale', max_iter=iteration)
svmClassNegative.fit(train_data, targatForClassNegative)
        
# Class 1 vs Class 2
targetForClassPositve = []
targetForClassPositve = [1 if train_target[i] == 1 else 0 for i in range(len(train_target))]
svmClassPositive = svm.SVC(kernel='linear', gamma='scale', max_iter=iteration)
svmClassPositive.fit(train_data, targetForClassPositve)

# For test from request
def svm_test(data):
    if svmClassNegative.predict([data]) == 1:
        return 0
    else:
        if svmClassPositive.predict([data]) == 1:
            return 1
        else:
            return 2
