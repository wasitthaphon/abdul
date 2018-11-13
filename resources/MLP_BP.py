from random import random
from random import seed
import math
import matplotlib.pyplot as plt
import os
import cfg
cwd = cfg.get_path()
import json

# Create 2 network layers input -> hidden -> output
def initialize_network(n_input, n_hidden_1, n_hidden_2,  n_outputs):
    network = list()
    hidden_layer_1 = [{'weights':[random() for i in range(n_input + 1)]} for i in range(n_hidden_1)]
    network.append(hidden_layer_1)
    hidden_layer_2 = [{'weights':[random() for i in range(n_hidden_1 + 1)]} for i in range(n_hidden_2)]
    network.append(hidden_layer_2)
    output_layer = [{'weights':[random() for i in range(n_hidden_2 + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Test the network
# seed(1)
# network = initialize_network(2, 1, 2)
# for layer in network:
#     print(layer)

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

# row = [1, 0, None]
# output = forward_propagate(network, row)
# print(output)

# 3 Back Propagate Error
# 3.1 Transfer Derivative
def transfer_derivative(output):
    return output * (1.0 - output)

# 3.2 Error Backpropagation
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# test backpropagation of error
# network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# 		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
# expected = [0, 1]
# backward_propagate_error(network, expected)
# for layer in network:
# 	print(layer)

# 4 Train Network
# 4.1 Update Weights
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# 4.2 Train Network
E = []
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        sum_error /= float(len(expected))
        if epoch % 100 == 0 and epoch != 0:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        E.append(sum_error)


# Predict
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Accuracy
def accuracy(actual, predicted):
    count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1
    return count / float(len(actual)) * 100

# Test training backprop algorithm
dataset = []
with open(cwd + 'sentensesData', 'r') as dataset_file:
    for data in dataset_file:
        tmp = []
        if '\n' in data:
            tmp = data[:-1]
        tmp = tmp.split('\t')
        for i in range(len(tmp)-1):
            tmp[i] = float(tmp[i])
        tmp[len(tmp)-1] = int(tmp[len(tmp)-1])
        dataset.append(tmp[0:])

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 15, 15, n_outputs)
learning_rate = 0.05
epoch = 10000
train_network(network, dataset, learning_rate, epoch, n_outputs)
with open(cwd + 'weights', 'w') as outfile:
    json.dump(network, outfile)

# for layer in network:
# 	print(layer)

expected = []
actual = []
for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
    actual.append(row[-1])
    expected.append(prediction)

print(accuracy(actual, expected))

plt.plot([i for i in range(epoch)], E, 'r')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()