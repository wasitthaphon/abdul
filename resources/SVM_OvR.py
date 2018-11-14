from sklearn import svm
from sklearn import metrics
from sklearn import multiclass
from sklearn import datasets
from sklearn import preprocessing
import math
import json
import cfg
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
train_data = preprocessing.scale(train_data)
# Test data
test_data = train_data.copy()
test_target = train_target.copy()
 
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

# Predict data
predicted = []
for data in test_data:
    if svmClassNegative.predict([data]) == 1:
        predicted.append(0)
    else:
        if svmClassPositive.predict([data]) == 1:
            predicted.append(1)
        else:
            predicted.append(2)

# Accuracy
count = 0
for i in range(len(predicted)):
    # print(predicted[i], test_target[i])
    if predicted[i] == test_target[i]:
        count += 1

accuracy = count / float(len(predicted)) * 100
print(accuracy)




# Plot
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

X = []
X0 = []
X1 = []
for data in train_data:
    X.append([data[1] - data[2], data[3]])
    X0.append(data[1] - data[2])
    X1.append(data[3])
y = (targatForClassNegative.copy(), targetForClassPositve.copy())


xx, yy = make_meshgrid(np.asarray(X0), np.asarray(X1))

models = (svmClassNegative.fit(X, targatForClassNegative),
        svmClassPositive.fit(X, targetForClassPositve))
titles = ('SVC Negative V Rest',
            'SVC Positive V Rest')
fig, sub = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.7, hspace=0.7)

for clf, title, ax, y_each in zip(models, titles, sub.flatten(), y):
    plot_contours(ax, clf, xx, yy,
                cmap=plt.cm.register_cmap(name='spectral', cmap='nipy_spectral'), alpha=0.8)
    ax.scatter(X0, X1, c=y_each, cmap=plt.cm.register_cmap(name='spectral', cmap='nipy_spectral'), s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Different')
    ax.set_ylabel('Point')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
# plt.show()

file_name = 'SVM_OvR_img.jpg'
plt.savefig(cfg.get_image_path() + file_name)

with open(cwd + 'info', 'r') as info:
    tmp = json.load(info)

svmJson = [{'SVM':[{'expected':predicted}, {'actual':test_target}, {'accuracy':accuracy}]}]
tmp.append(svmJson)

with open(cwd + 'info', 'w') as info:
    json.dump(tmp, info)