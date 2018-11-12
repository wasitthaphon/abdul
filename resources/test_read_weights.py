import os
import json
cwd = os.getcwd() + '/resources/'

network = []
with open(cwd + 'weights', 'r') as f:
    network = json.load(f)

with open(cwd + 'max_sentense', 'r') as mx:
    for m in mx:
        print(float(m))