# neural network  
# every neuron recieves input from all the previous neurons
# every neuron has as many weights as it has inputs
#		+ one more for the bias term 

import numpy as np
from math import exp
import random
from pprint import pprint

random.seed(42)

x = [[0,0,1], [1,1,1], [1,0,1], [0,1,1]]
y = [[0],[1],[1],[0]]

def sigmoid(signal):
  return 1/(1+exp(-signal))

def sigmoid_derivative(signal):
  return sigmoid(signal) * (1-sigmoid(signal))

def least_squares_loss(pred,target):
  if type(pred) != type([]):
    pred = [pred]
  if type(target) != type([]):
    target = [target]
  sub = 0
  for p,t in zip(pred,target):
    sub += t-p
  return (sub)**2

def numerical_derivative(func, func_input, respect_to_index=0, h=0.0001):
  """Compute the numerical derivative of a function
  Args:
    func (function): A function
    func_input (list): A list of inputs given to the function
    respect_to_index (int): The index of the value the derivative is calculated with respect to
    h (float): the amount to tweak the function with to compute the gradient
  Returns:
    float: the derivative of the function at the input 
  """
  
  # compute function value
  f0 = func(*func_input)

  # compute inputs with slightly tweaked respective input weight
  new_respect_to_weight = np.add(func_input[respect_to_index], h)
  func_input_h = func_input
  func_input_h[respect_to_index] = new_respect_to_weight

  f0_h = func(*func_input_h)

  # compute the difference, the slope of the function
  delta_f = f0_h - f0
  delta_f_normalized = delta_f/h

  return delta_f_normalized

def neuron_output(inputs, weights, activation=sigmoid):
  signal = 0

  # add up all the output
  for x,weight in zip(inputs,weights[:-1]):
    signal += x * weight
  
  # compute output as the activation function to the signal plus the bias
  output = activation(signal) + weights[-1] 

  return output

def create_neuron(n_weights):
  n = {"weights" : [random.random() for _ in range(n_weights + 1)]}
  return n

def create_neurons(n_weights, n_neurons):
  neurons = []
  for i in range(n_neurons):
    n = create_neuron(n_weights)
    neurons.append(n)
  return neurons

def create_network(n_input, n_hidden, n_output, hidden_size=10):
  layers = []

  input_layer = create_neurons(n_input, hidden_size)
  layers.append(input_layer)

  for _ in range(n_hidden):
    hidden_layer = create_neurons(hidden_size, hidden_size)
    layers.append(hidden_layer)
  
  out_layer = create_neurons(hidden_size, n_output)
  layers.append(out_layer)

  return layers

def network_forward(network, inputs):
  """Computes the forward pass of a network 
  args:
    network: a network as created with the create network function
    inputs: a list consisting of the input features of a single example
  return: 
    a prediction for the given input
  """
  for layer in network:
    next_layer_inputs = []
    for neuron in layer: 
      output = neuron_output(inputs, neuron["weights"], sigmoid)
      neuron["out"] = output
      next_layer_inputs.append(output)
    inputs = next_layer_inputs
  
  return inputs

def network_back(network, target, loss=least_squares_loss):
  """Computer weight deltas using the chain rule. 
  """
  for i in reversed(range(len(network))):
    errors = []
    layer = network[i]
    if i == len(network)-1: # if it's the last layer
      for j in range(len(layer)):
        neuron = layer[j]
        error = loss(neuron["out"], target[j])
        errors.append(error)
    else:
      for j in range(len(layer)):
        neuron = layer[j]
        error = 0.0

        parent_layer = network[i+1]
        for k in range(len(parent_layer)): # neurons is parent layer
          parent_neuron = parent_layer[k]
          error += parent_neuron["weights"][j] * parent_neuron["delta"]
        errors.append(error)

    for j in range(len(layer)):
      neuron = layer[j]
      neuron["delta"] = errors[j] * numerical_derivative(sigmoid, [neuron["out"]])

#
x_sample = x[0]
y_sample = y[0]


net = create_network(3, 1, 1)

forward_pass = network_forward(net, x_sample)
print (forward_pass)

# backwards pass: 
loss = least_squares_loss(forward_pass, y_sample)
print (loss)

bp = network_back(net, y_sample)

for layer in net:
  for neuron in layer:
    pprint (neuron)

#print (numerical_derivative(sigmoid, forward_pass))