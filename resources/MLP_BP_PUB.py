from random import random
from random import seed
import math
import os
import json
cwd = os.getcwd() + '/resources/'

# Find amount bad and good word
def find_word(message):
    words = []
    with open(cwd + 'words', 'r') as words_file:
        for word in words_file:
            word = word.replace('\n', '')
            word = word.replace('\ufeff', '')
            word = word.split('\t')
            words.append(word)
    words.sort(key = lambda s : len(s[0]), reverse=True)

    max_sentense = 1
    with open(cwd + 'max_sentense', 'r') as max_sentense_file:
        for max_st in max_sentense_file:
            max_sentense = float(max_st)

    points = 0
    amount = [0, 0]
    wordtmp = [[], []]
    idx = []
    result = []
    for word in words:
        for j in range(len(message) - len(word[0]) + 1):
            string = message[j:j+len(word[0])]
            if word[0].__eq__(string) and not (j in idx):
                if float(word[1]) < 0:
                    amount[0] += len(word[0])
                    wordtmp[0].append(word[0])
                elif float(word[1]) > 0:
                    amount[1] += len(word[0])
                    wordtmp[1].append(word[0])
                points += float(word[1])
                for k in range(j, j + len(word[0])):
                    idx.append(k)
    amount[0] = float((amount[0] * 100.0) / len(message))
    amount[1] = float((amount[1] * 100.0) / len(message))
    result.append([len(message)/max_sentense, amount[0], amount[1], points, wordtmp])
    return result

# Neural Activation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Neural Transfer
def tranfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))

# Forward Propagation
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = tranfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Predict
def predict(network, row):
    outputs = forward_propagate(network, row)
    print(outputs)
    return outputs.index(max(outputs))

# Load Weights
def load_weights():
    network = []
    with open(cwd + 'weights', 'r') as weights:
        network = json.load(weights)
    return network