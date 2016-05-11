import numpy as np

from convertResult import *
from sigmoid import *

def costFunction(X, y, Theta, LAMBDA):
	""" Neural network cost function with 3 layers: 
			input layer -> hidden layer -> output layer
			
		The formula is in ex4.pdf page(5)
		
		LAMBDA: LAMBDA for regularization
		
		X: m x 400 (m is #training cases)
		y: m x 1 
		
		Input layer (layer 1): X: m x 400
		Hidden layer (layer 2): 400 x 25
		Output layer (layer 3): 25 x 10
		
		Therefore:
			Theta1: 25 x 401 (Add one more column with value 1)
			Theta2: 10 x 26 (Add one more column with value 1)
	"""
	# split Theta into Theta1 and Theta2
	Theta1 = np.reshape(Theta[:25 * 401], (25, 401), order='F').copy()
	Theta2 = np.reshape(Theta[25 * 401:], (10, 26), order='F').copy()
	
	m = X.shape[0]
	
	# Add one more column with value 1 to X
	newX = np.append(np.ones((m, 1)), X, 1) # newX (m x 401)
	
	newY = convertNumberToArray(y) # newY (m x 10)
		
	# Feed forward
	
	# Hidden layer (layer 2)
	z2 = newX.dot(Theta1.T) # z2 (m x 25)
	a2 = sigmoid(z2) 
	
	# Add one more column with value 1 to a2
	a2 = np.append(np.ones((a2.shape[0], 1)), a2, 1) # a2 (m x 26)
	
	# Output layer (layer 3)
	z3 = a2.dot(Theta2.T) # z3 (m x 10)
	a3 = sigmoid(z3) # a3 (m x 10)
	
		
	# Calculate cost function (use logistic regression cost function)
	J = np.log(a3) * newY + np.log(1 - a3) * (1 - newY)
	J = -np.sum(J) / m;
	
	# Back propagation
	
	sigma3 = a3 - newY # sigma3 (m x 10)
	
	# Theta2 (10 x 26), sigma3 (m x 10), z2 (m x 25)
	sigma2 = sigma3.dot(Theta2[:, 1:]) * sigmoidGradient(z2) # sigma2 (m x 25)
	
	Theta2_grad = sigma3.T.dot(a2) # Theta2_grad (10 x 26)
	Theta1_grad = sigma2.T.dot(newX) # Theta1_grad (25 x 401)
	
	# Add regularization (ex4 page 6)
	J += LAMBDA * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2)) / (2 * m)
	Theta1_grad[:, 1:] += LAMBDA / m * Theta1[:, 1:]
	Theta2_grad[:, 1:] += LAMBDA / m * Theta2[:, 1:]
	
	# Put Theta1_grad and Theta2_grad to grad
	grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))
	
	return (J, grad, a3)