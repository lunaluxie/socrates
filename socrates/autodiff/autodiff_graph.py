"""Graph based autodiff

Supports two modes
    - Forward mode
    - Reverse mode (much more efficient)
        We use reverse mode
        Yet the graph method is still inefficient
"""
import numpy as np

class Expression():
    def __init__(self, value):
        self.value = value
        self.dependencies = []
        self.grad_value = None
    
    def __add__(self, other):
        z = Variable(self.value + other.value)
        self.dependencies.append((1.0, z))
        other.dependencies.append((1.0, z))
        return z

    def __mul__(self, other):
        z = Variable(self.value * other.value)
        self.dependencies.append((other.value, z))
        other.dependencies.append((self.value, z))
        return z

class Variable(Expression):
    def __init__(self, value):
        Expression.__init__(self, value)

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.dependencies)
        return self.grad_value

class Constant(Expression):
    def __init__(self, value):
        Expression.__init__(self, value)
        self.grad_value = 1

    def grad(self):
        return self.grad_value


def sin(x):
    z = Variable(np.sin(x.value))
    x.dependencies.append((np.cos(x.value), z))
    return z

def grad(x):
    """Computes the gradient with respect to variable
    
    Args:
        x (Variable): Variable with respect to which to compute the gradient
    """

    def deepest_dependency_grad_value(x):
        for dependency in x.dependencies:
            var = dependency[1]
            if not var.dependencies:
                var.grad_value = 1
                return
            else:
                deepest_dependency_grad_value(var)
    
    deepest_dependency_grad_value(x)

    return x.grad()

x = Variable(np.array([5,4,2]))
#y = Variable(np.array([1,2,3]))
#z = x * y + x
f = sin(x)

print (grad(x))

# Downside:
# We have to reset the graph after each time
# we use it. 