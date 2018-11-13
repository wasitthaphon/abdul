import os
import json
import cfg

cwd = cfg.get_path()

network = []
with open(cwd + 'weights', 'r') as f:
    network = json.load(f)

with open(cwd + 'max_sentense', 'r') as mx:
    for m in mx:
        print(float(m))