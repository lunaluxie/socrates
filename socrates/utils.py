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

def numerical_definite_integral(func, interval, n=5000):
	"""Find the numerical value of of the definite integral 
	with arbritrary precision using the rectangle method.
	Args:
    func (function): A function with one integer like input argument
		interval (list of float): length 2 denotes the start and stop value for the function.  
    n (integer): Precision with which to calculate the  
  Returns:
    float: The definite integral in the interval
	"""
	delta_x = (interval[1]-interval[0])/n
	area = 0
	# choose middle point in each sub-interval
	cursor_loc = interval[0] + 0.5*delta_x
	for rectangle in range(n):
		rect_height = func(cursor_loc)

		# add the area
		rect_area = delta_x * rect_height
		area += rect_area

		# update cursor location
		cursor_loc += delta_x
	return area