# Socrates
> Sane implementation of machine learning algorithms

## Installation 
```
$ git clone https://github.com/Kasperfred/socrates
$ cd socrates
$ pip install .
```
## Basic usage
Define a neural network with 2 input neurons, 1 hidden layer with 10 neurons, and an output layer with 1 neuron, and run backpropagation once to find the new weights. Finally, we print these weights.

```Python
from socrates.nn import *

x = [0,0,1]
y = [0]

# define network
net = create_network(3, 1, 1)

forward_pass = network_forward(net, x_sample)

# backwards pass using least squares loss
loss = least_squares_loss(forward_pass, y_sample)
back_prop = network_back(net, y_sample)

# print the new weights
print (numerical_derivative(sigmoid, forward_pass))
```
Note this is just an example, and the API will probably change in the future making this unusable.