import numpy as np 
import os
import cfg

f = open(cfg.get_path() + 'words', 'w')

words = []
chk = []
cwd = cfg.get_path()
first = True
with open(cwd + 'normal-word.txt', 'r') as norWord:
    for item in norWord:
        item = item.replace('\n', '')
        if not item in chk:
            words.append([item, 100 * np.random.uniform(0, 0.05)])
            chk.append(item)

first = True
with open(cwd + 'negative-word', 'r') as negWord:
    for item in negWord:
        # print(item[0:len(item)-1])
        item = item.replace('\n', '')
        if not item in chk:
            words.append([item, (-100 * np.random.uniform(0.5, 0.01))])
            chk.append(item)

first = True
with open(cwd + 'positive-sentiment-words.txt', 'r') as posWord:
    for item in posWord:
        item = item.replace('\n', '')
        if not item in chk:
            words.append([item, (100 * np.random.uniform(0.2, 0.6))])
            chk.append(item)

first = True
with open(cwd + 'swear-words.txt', 'r') as sweWord:
    for item in sweWord:
        item = item.replace('\n', '')
        if not item in chk:
            words.append([item, (-100 * np.random.uniform(0.7, 1))])
            chk.append(item)


print(len(words))
for item in words:
    # print(item)
    
    f.write('{}\t{}\n'.format(item[0], item[1]))
f.close()