"""Tape-based autodifferentiation"""

"""In general there are two data structure that can be used for autograd
Graph based, and vector (tape) based. Tapebased keeps everything neatly together
and is preferable for a range of reasons. 
"""

class Node():
    def __init__(self, weights, dependencies):
        if not type(weights) == list:
            weights = [weights]
        if not type(dependencies) == list:
            dependencies = [dependencies]
        
        self.weights = weights
        self.dependencies = dependencies # parents

class Tape():
    def __init__(self):
        self.nodes = []
    
    def add(self, node):
        index = len(self.nodes)
        self.nodes.append(node)
        return index

def Variable():
    def __init__(self, value, tape, index=None):
        self.value = value
        if index:
            self.index = index
        else:
            self.index = tape.add(Node(1,0))
    
    def __mul__(self, other):
        pass

