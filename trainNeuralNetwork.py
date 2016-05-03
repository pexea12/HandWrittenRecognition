import numpy as np

from costFunction import *
from scipy.optimize import minimize

def trainNeuralNetwork(X, y, Theta1, Theta2, LAMBDA):
	# Put Theta1 and Theta2 to Theta
	Theta = np.hstack((Theta1.T.ravel(), Theta2.T.ravel()))
	
	costFunc = lambda p: costFunction(X, y, p, LAMBDA)[0]
	gradFunc = lambda p: costFunction(X, y, p, LAMBDA)[1]
	
	result = minimize(costFunc, Theta, method='CG', jac=gradFunc, options={ 'disp': True, 'maxiter': 50.0 })
	
	Theta = result.x
	Theta1 = np.reshape(Theta[:25 * 401], (25, 401), order='F').copy()
	Theta2 = np.reshape(Theta[25 * 401:], (10, 26), order='F').copy()
	
	J, _, a3 = costFunction(X, y, Theta, LAMBDA)
	
	return J, Theta1, Theta2, a3