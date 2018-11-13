import os
import cfg

items = []
with open(cfg.get_path() + '131161_sentense', 'r') as sentenses_file:
    for sentense in sentenses_file:
        sentense = sentense.replace('\n', '')
        sentense = sentense.replace('\ufeff', '')
        sentense = sentense.split('-')
        items.append(sentense)

# Check
print(items[0])
print(len(items))

stns_file = open(cfg.get_path() + 'sentenses', 'a')
answ_file = open(cfg.get_path() + 'answer', 'a')
comparator = ['บวก', 'ลบ']

for item in items:
    ans = 2
    if comparator[0] in item[1]:
        ans = 1
    elif comparator[1] in item[1]:
        ans = 0
    else:
        ans = 2
    answ_file.write('{}\n'.format(ans))
    stns_file.write('{}\n'.format(item[0]))
