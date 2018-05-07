"""Graph based autodiff

Supports two modes
    - Forward mode
    - Reverse mode (much more efficient)
        We use reverse mode
        Yet the graph method is still inefficient
"""
import math

class Variable:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    def __add__(self, other):
        z = Variable(self.value + other.value)
        self.children.append((1.0, z))
        other.children.append((1.0, z))
        return z

    def __mul__(self, other):
        z = Variable(self.value * other.value)
        self.children.append((other.value, z))
        other.children.append((self.value, z))
        return z

def sin(x):
    z = Variable(math.sin(x.value))
    x.children.append((math.cos(x.value), z))
    return z

x = Variable(10)
y = Variable(9)
z = x * y + sin(x) * x

z.grad_value = 1
print (x.grad())

def grad(x):
    """Computes the gradient with respect to variable
    
    Args:
        x (Variable): Variable with respect to which to compute the gradient
    """

    def deepest_child_grad_value(x):
        for child in x.children:
            if not child.children:
                child.grad_value = 1
                return
            else:
                deepest_child_grad_value(child)
    
    return x.grad()

print (grad(x))

# Downside:
# We have to reset the graph after each time
# we use it. 