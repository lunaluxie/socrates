import numpy as np

def least_squares_loss(pred,target):
	"""Euclidian distance between a target value and predicted value
	Args:
		pred (array): matrix of predicted value
		target (array): matrix of target prediction
	Returns: 
		float: ecludian error between pred and target
	"""

	# type checking
  if type(pred) != type([]):
    pred = [pred]
  if type(target) != type([]):
    target = [target]
	
	# compute error
  sub = 0
  for p,t in zip(pred,target):
    sub += t-p
  error = (sub)**2

	return error