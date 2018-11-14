# dataset = []
# with open('sentensesData', 'r') as data_file:
#     for data in data_file:
#         if '\n' in data:
#             data = data[:-1]
        
#         tmp = data.split('\t')
#         for i in range(len(tmp)-1):
#             tmp[i] = float(tmp[i])
#         tmp[len(tmp)-1] = int(tmp[len(tmp)-1])

#         dataset.append(tmp)
    
# print(dataset)

# ans = []
# with open('answer', 'r') as ans_file:
#     for data in ans_file:
#         tmp = 0
#         if '\n' in data:
#             tmp = data[:-1]
#         ans.append(int(tmp))
# print(ans)

import json
import cfg

tmp = []
with open(cfg.get_path() + 'info', 'r') as fl:
    tmp = json.load(fl)

print(tmp[0][0]['MLP-BP'][0])
print(tmp[1])