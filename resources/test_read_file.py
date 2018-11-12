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

ans = []
with open('answer', 'r') as ans_file:
    for data in ans_file:
        tmp = 0
        if '\n' in data:
            tmp = data[:-1]
        ans.append(int(tmp))
print(ans)
