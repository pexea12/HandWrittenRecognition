import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from sigmoid import *
from displayData import *
from costFunction import *
from convertResult import *
from trainNeuralNetwork import * 
from randInitializeWeights import *

# Load data from ex4data1.mat
data = sio.loadmat('ex4data1.mat')

X = data['X']
y = data['y']

# X(5000 x 400) - 5000 cases, each case is a 20x20-pixel image

# Select 100 random tests to view items

select_value = np.arange(5000)
np.random.shuffle(select_value)

X_random = X[select_value[0:100]]

# Visualize the data. 

displayData(X_random)
input('Press any key to continue\n')



# Set regularization parameter LAMBDA = 1
LAMBDA = 1
X = np.array(X)
y = np.array(y)

# Randomly initialize Theta1, Theta2
Theta1 = randInitializeWeights(400, 25)
Theta2 = randInitializeWeights(25, 10)

# Start training with 100 iterations ...
J, Theta1, Theta2, a3 = trainNeuralNetwork(X, y, Theta1, Theta2, LAMBDA)

# With the trained Theta1, Theta2, try to predict: 

print("J = ", J)

a3 = np.round(a3)
result = convertArrayToNumber(a3)

accuracy = np.sum(result == y)

print('Accuracy: ', accuracy / 5000 * 100, '%')

# Select 5 random training test and try to recognize them with sample Theta1 and Theta2

curr_select_value = select_value[0:5]

for i in range(5):
	print('Test ', i, ': ')
	print('\tDisplay: ')
	displayData(np.array([X[curr_select_value[i]]]))
	print('\tSample result: ', y[curr_select_value[i]] % 10)
	print('\tPredict result: ', result[curr_select_value[i]] % 10)
	input('Press any key to continue')
	
