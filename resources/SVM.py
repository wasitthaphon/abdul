from sklearn import svm

dataset = []
target = []
with open('sentensesData', 'r') as dataset_file:
    for data in dataset_file:
        tmp = []
        if '\n' in data:
            tmp = data[:-1]
        tmp = tmp.split('\t')
        for i in range(len(tmp)-1):
            tmp[i] = float(tmp[i])
        tmp[len(tmp)-1] = int(tmp[len(tmp)-1])
        dataset.append(tmp[:len(tmp)-2])
        target.append(tmp[len(tmp)-1])

clf = svm.SVC(gamma='scale', max_iter=1500)
clf.fit(dataset, target)
print(clf.score(dataset, target) * 100)

