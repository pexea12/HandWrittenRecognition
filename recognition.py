import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
	def __init__(self, nLayers, nIteration=10, learningRate=0.1, regularization=0.1):
		### nLayers: number of layers
		### nIteration: number of iterations
		self.nLayers = nLayers
		self.nIteration = nIteration
		self.learningRate = learningRate
		self.regularization = regularization

	def init(self, X_train, y_train):
		### self.nLayer: number of neural at each layer (28*28, 30, 10)
		### self.value: value of neural at each layer
		### self.w: Theta (30, 28*28 + 1 bias unit) and (10, 30 + 1 bias unit)
		### X_train: add 1 more bias column
		### y_train
		
		self.nLayer = [int(X_train.shape[1]), 30, 10]
		
		self.m = X_train.shape[0]
		self.X_train = np.append(np.ones((self.m, 1)), X_train, axis=1)
		self.y_train = y_train
		self.y_train = self.makeOutput()
		
		self.value = [
			np.zeros((self.m, self.nLayer[0] + 1)),
			np.zeros((self.m, self.nLayer[1] + 1)),
			np.zeros((self.m, self.nLayer[2]))
		]
			
		self.errorLocal = [
			np.zeros((self.m, self.nLayer[1])),
			np.zeros((self.m, self.nLayer[2]))
		]
		
		self.w = [
			np.random.uniform(-0.12, 0.12, (self.nLayer[1], self.nLayer[0] + 1)),
			np.random.uniform(-0.12, 0.12, (self.nLayer[2], self.nLayer[1] + 1))
		]
		
		self.cost = 0

	def makeOutput(self):
		out = np.zeros((y_train.shape[0], 10))
		for i in range(y_train.shape[0]):
			out[i, y_train[i]] = 1
		return out

	def feedForward(self):
		self.value[0] = self.X_train
		self.value[1] = self.sigmoid(self.value[0].dot(self.w[0].T))
		self.value[1] = np.append(np.ones((self.m, 1)), self.value[1], axis=1)
		self.value[2] = self.sigmoid(self.value[1].dot(self.w[1].T))

	### back propagation
	def backPropa(self):
		self.errorLocal[1] = self.value[2] - self.y_train
		self.errorLocal[0] = self.errorLocal[1].dot(self.w[1][:, 1:]) * self.sigmoid_derivative(self.value[0].dot(self.w[0].T))

		self.w[1] -= self.learningRate * self.errorLocal[1].T.dot(self.value[1])
		self.w[0] -= self.learningRate * self.errorLocal[0].T.dot(self.X_train)
		
		# add regularization
		self.w[1][:, 1:] -=  self.regularization / self.m * self.w[1][:, 1:]
		self.w[0][:, 1:] -=  self.regularization / self.m * self.w[0][:, 1:]

	def train(self, X_train, y_train):
		self.init(X_train=X_train, y_train=y_train)
		
		for i in range(0, self.nIteration):
			print(i, ':')
			self.feedForward()
			self.backPropa()
			print('Cost = ', self.costFunction())


	def costFunction(self):
		# not add regularization yet
		J = np.log(self.value[2]) * self.y_train + np.log(1 - self.value[2]) * (1 - self.y_train)
		J = -np.sum(J) / self.m
		
		# add regularization
		J += self.regularization * (np.sum(self.w[0][:, 1:] ** 2) + np.sum(self.w[1][:, 1:] ** 2)) / (2 * self.m)
		
		return J

	def classification(self, X_test, y_test):
		m = X_test.shape[0]

		X_test_temp = np.append(np.ones((m, 1)), X_test, axis=1)
		test_result = self.sigmoid(X_test_temp.dot(self.w[0].T))
		test_result = np.append(np.ones((m, 1)), result, axis=1)
		test_result = self.sigmoid(result.dot(self.w[1].T))
		
		test_result = np.round(test_result)
		result = np.zeros((m, 1))

		for i in range(self.m):
			for j in range(0, 10):
				if (self.value[2][i, j] == 1):
					result[i] = j
		
		accuracy = np.sum(result == y_test)
		print('Accuracy = ', accuracy)


	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoid_derivative(self, z):
		sigVal = self.sigmoid(z)
		return sigVal * (1 - sigVal)


data = pd.read_csv("mnist_train_100.csv", header=None)
X_train = data.iloc[:, 1:].values # X_train(100 x 784)
X_train.astype(float)
y_train = data.iloc[:, 0].values # y_train(100 x 1)

### vẽ lại
"""
ig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
	img = X_train[y_train == i][0].reshape(28, 28)
	ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""

X_train = X_train / 256
NN = NeuralNetwork(nLayers=3, nIteration=1000, learningRate=0.01)

# print(y_train)
NN.train(X_train, y_train)
