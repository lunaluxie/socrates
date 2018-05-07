class Network:
	def __init__(self, layers=None, loss=None):
		self.layers = layers
		self.loss = loss

	def _self_check(self):
		"""Check the integrity of the instance variables
		"""
		if not self.layers:
			raise Exception("Selfcheck failed: No layers were found")
		
		if not self.loss:
			raise Exception("Selfcheck failed: No loss function was found")

	def add(self, layer):
		"""Add a single layer object of the layer class
		"""
		self.layers.append(layer)
		return self.layers
	
	def add_many(self, layers):
		"""Add a list of layer objects of the layer class
		"""
		self.layers.extend(layers)
		return self.layers
	
	def get_layers(self):
		return self.layers
	
	def forward(self):
		pass