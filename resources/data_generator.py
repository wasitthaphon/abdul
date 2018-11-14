import os
import cfg
import statistics

wordSet = []
sentensesData = []
cwd = cfg.get_path()

with open(cwd + 'words', 'r') as words:
    for word in words:
        # print(word)
        word = word.split('\t')
        word[0] = word[0].replace('\ufeff', '')
        word[1] = word[1].replace('\n', '')
        wordSet.append(word)

with open(cwd + 'sentenses', 'r') as sentenses:
    for sentense in sentenses:
        sentense = sentense.replace('\n', '')
        sentensesData.append(sentense)

wordSet.sort(key=lambda s:len(s[0]), reverse=True)
# print(wordSet)
newSentensesData = []
max_sentense = 0
for sentense in sentensesData:
    if len(sentense) > max_sentense:
        max_sentense = len(sentense)

for sentense in sentensesData:
    points = 0
    amount = [0, 0]
    wordtmp = [[], []]
    idx = []
    
    for word in wordSet:
        for j in range(len(sentense) - len(word[0])):
            string = sentense[j:j+len(word[0])]
            if word[0].__eq__(string) and not j in idx:

                # Bad word
                if float(word[1]) < 0:
                    amount[0] += len(word[0])
                    wordtmp[0].append(word[0])
                # Good word
                elif float(word[1]) > 0:
                    amount[1] += len(word[0])
                    wordtmp[1].append(word[0])
                points += float(word[1])
                for k in range(j, j + len(word[0]) + 1):
                    idx.append(k)
    newSentensesData.append([sentense, len(sentense), amount[0], amount[1], points, wordtmp])

answer = []
with open(cwd + 'answer', 'r') as ans_file:
    for data in ans_file:
        tmp = 0
        if '\n' in data:
            tmp = data[:-1]
        answer.append(int(tmp))

f = open(cwd + 'sentensesDataWatch', 'w')
l = open(cwd + 'sentensesData', 'w')
g = open(cwd + 'max_sentense', 'w')
# print(sentensesData)
i = 0
for data in newSentensesData:
    # Store data SENTENCE, SENTENCE_LENGHT, NEGATIVE_WORD_AMOUNT, POSITIVE_WORD_AMOUNT, POINTS, WORDS(OPTIONAL)
    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data[0], data[1], data[2], data[3], data[4], data[5],answer[i]))
    l.write('{}\t{}\t{}\t{}\t{}\n'.format(data[1], data[2], data[3], data[4], answer[i]))
    i += 1
g.write(str(max_sentense))
g.close()
f.close()
l.close()    
