# neural network  
# every neuron recieves input from all the previous neurons
# every neuron has as many weights as it has inputs
#		+ one more for the bias term 

import numpy as np
from math import exp
import random
from pprint import pprint

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

"""
New FILE
"""

from typing import Callable
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

def numerical_derivative(func: Callable, func_input: list, respect_to_index: int=0, h: float=0.0001) -> float:
  """Compute the numerical derivative of a function

  Args:
    func (Callable): Function of which to compute the derivative
    func_input (list): Inputs to the function
    respect_to_index (int, optional): Defaults to 0.
    h (float, optional): Defaults to 0.0001. The stepsize

  Returns:
    float: The derivative at the input point with `h` stepsize
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
  """Compute weight deltas using chainrule
  
  Args:
    network ([type]): [description]
    target ([type]): [description]
    loss ([type], optional): Defaults to least_squares_loss. [description]
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

network_back(net, y_sample)

for layer in net:
  for neuron in layer:
    pprint (neuron)

#print (numerical_derivative(sigmoid, forward_pass))

"""NEW FILE
"""

feature_samples = np.array([ [x,y] for x,y in zip(range(1,100), range(1,100)) ])
answer_samples = np.array([ 2*x - 3*y for x,y in feature_samples ])
learning_rate = 0.1

def sigmoid(z):
	return 1/(1+np.exp(-z))

def dsigmoid(z):
	return np.multiply(z, np.subtract(1,z))

def relu(x):
	return max(0,x)

def loss(x,y):
	return (x-y)**2

# weights
# or
theta1 = np.array([ [3, 5] ])
bias1 = np.array([5])

# not
theta2 = np.array([ [3] ])
bias2 = np.array([5])

# not (faulty)
theta3 = np.array([ [0.1] ])
bias3 = np.array([2])

for _ in range(40):
	errs = [0,0,0]
	for x,y in zip(feature_samples, answer_samples):

		# forward prop
		z1 = np.matmul(theta1, x) + bias1
		a1 = [sigmoid(z_) for z_ in z1]

		z2 = np.matmul(theta2, a1) + bias2
		a2 = [sigmoid(z_) for z_ in z2]

		z3 = np.matmul(theta3, a2) + bias3
		a3 = [sigmoid(z_) for z_ in z3]

		# backprop

		# total error and blame for last layer
		error_layer_3 = loss(a3, y)

		# compute blame for layer 2
		g2_prime = dsigmoid(a2) # sigmoid derivative

		# g2_prime_numerical_approx = numerical_derivative(sigmoid, [z2])
		#      quite expensive, but universal
		#      confirms the analytical derivative

		error_layer_2 = np.multiply(np.matmul(theta2.T, error_layer_3), g2_prime)

		# compute blame for layer 1
		g1_prime = dsigmoid(a1)
		error_layer_1 = np.multiply(np.matmul(theta3.T, error_layer_2), g1_prime)

		# accumilate errors
		errs[0] += np.matmul(error_layer_1, np.transpose(a1))
		errs[1] += np.matmul(error_layer_2, np.transpose(a2))
		errs[2] += np.matmul(error_layer_3, np.transpose(a3))

	print (errs)
	num_samples = len(feature_samples)

	# get the actual gradient for layer 1
	gradient_layer_1 = np.add(np.multiply(1/num_samples, errs[0]), theta1)
	gradient_layer_1_bias = np.multiply(1/num_samples, errs[0])
	# print (gradient_layer_1, gradient_layer_1_bias)


	# get the actual gradient for layer 2
	gradient_layer_2 = np.add(np.multiply(1/num_samples, errs[1]), theta2)
	gradient_layer_2_bias = np.multiply(1/num_samples, errs[1])
	# print (gradient_layer_2, gradient_layer_2_bias)

	# get the actual gradient for layer 3
	gradient_layer_3 = np.add(np.multiply(1/num_samples, errs[2]), theta3)
	gradient_layer_3_bias = np.multiply(1/num_samples, errs[2])
	# print (gradient_layer_3, gradient_layer_3_bias)


	# update weights

	# weights
	# or
	theta1 = np.subtract(theta1, np.multiply(gradient_layer_1, learning_rate))
	bias1 = np.subtract(bias1, np.multiply(gradient_layer_1_bias, learning_rate))

	# not
	theta2 = np.subtract(theta2, np.multiply(gradient_layer_2, learning_rate))
	bias2 = np.subtract(bias2, np.multiply(gradient_layer_2_bias, learning_rate))

	# not (faulty)
	theta3 = np.subtract(theta3, np.multiply(gradient_layer_3, learning_rate))
	bias3 = np.subtract(bias3, np.multiply(gradient_layer_3_bias, learning_rate))

print (theta1, bias1)
print (theta2, bias2)
print (theta3, bias3)


for x in feature_samples[:10]:
	z1 = np.matmul(theta1, x) + bias1
	a1 = [sigmoid(z_) for z_ in z1]

	z2 = np.matmul(theta2, a1) + bias2
	a2 = [sigmoid(z_) for z_ in z2]

	z3 = np.matmul(theta3, a2) + bias3
	a3 = [sigmoid(z_) for z_ in z3]
	print (x, a3)
