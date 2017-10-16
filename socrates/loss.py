class ProtoLoss:
	def __init__(self):
		pass

	def __str__(self):
		return "Abstract loss function"

	def loss(self):
		pass
	
	def loss_derivative(self):
		pass 

pl = ProtoLoss()

print (pl)