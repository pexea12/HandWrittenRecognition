import numpy as np

def randInitializeWeights(L_in, L_out):
	"""Randomly initialize Theta to break symmetry
	   L_in: Number of input
	   L_out: Number of output
	   
	   return Theta is maxtrix L_out x (L_in + 1) (L_in + 1 because we add one more column with value 1)
	"""
	epsilon_init = 0.12
	
	return np.random.random((L_out, L_in + 1)) * 2 * epsilon_init - epsilon_init