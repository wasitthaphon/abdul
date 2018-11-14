from sklearn import svm
from sklearn import metrics
from sklearn import multiclass
from sklearn import datasets
import math
import json
import cfg

cwd = cfg.get_path()
dataset = []
target = []
with open(cwd + 'sentensesData', 'r') as dataset_file:
    for data in dataset_file:
        tmp = []
        data = data.replace('\n', '')
        data = data.replace('\ufeff', '')
        tmp = data.split('\t')

        for i in range(len(tmp)-1):
            tmp[i] = float(tmp[i])
        tmp[len(tmp)-1] = int(tmp[len(tmp)-1])
        dataset.append(tmp[:len(tmp)-1])
        target.append(tmp[-1])

mesT = []
epoch = 5000

def sigmoid(actual):
    tmp = []
    for item in actual:
        tmp.append(1.0 / (1.0 + math.exp(-item)))
    return tmp


model = svm.SVC(kernel='sigmoid', gamma='scale', max_iter=2000, C=15, probability=True)
clf = multiclass.OneVsRestClassifier(model).fit(dataset, target)
print(actual)
y_pred = clf.predict(dataset)

scores = clf.score(dataset, target)

# with open(cwd + 'info', 'r') as info:
#     tmp = json.load(info)
# tmp.append([{'SVM':[{'expected':target}, {'actual':y_pred.tolist()}, {'accuracy':scores}]}])

# with open(cwd + 'info', 'w') as info:
#     json.dump(tmp, info)