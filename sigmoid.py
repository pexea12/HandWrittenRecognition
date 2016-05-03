import numpy as np

def sigmoid(z):
	"""Calculate sigmoid function"""
	return 1 / (np.exp(-z) + 1)

def sigmoidGradient(z):
	"""Calculate sigmoid gradient function: Dao ham cua ham sigmoid"""
	"""g'(z) = g(z) * (1 - g(z))"""
	return sigmoid(z) * (1 - sigmoid(z))