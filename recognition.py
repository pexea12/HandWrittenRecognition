import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("mnist_train_100.csv", header = None)
y_train = data.iloc[:, 0].values

X_train = data.iloc[:, 1:].values

ig, ax = plt.subplots(nrows=2, ncols=5, sharex=True,
sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

class NeuralNetwork:
    def __init__(self, nLayers, nInteration = 10, learningRate = 0.1):
        self.nLayers = nLayers
        self.nInteration = nInteration
        self.learningRate = learningRate

    def initWeight(self):
        self.w = [np.random.uniform(-1.0, 1.0, (self.nLayer[0] + 1, self.nLayer[1])),
                np.random.uniform(-1.0, 1.0, (self.nLayer[1] + 1, self.nLayer[2]))]

    def fitData(self, X_train, y_train):
        self.nLayer = [int(X_train.shape[1]), int(30), 10]
        self.value = [np.zeros(self.nLayer[0]) ,np.zeros(30), np.zeros(10)]
        self.initWeight()

        for _ in range(self.nInteration):
            for nOfSet in range(X_train.shape[0]):
                self.value[0] = X_train[nOfSet]
                for i in range(1, self.nLayers):
                    s = self.netInput(self.value[i-1], self.w[i-1])
                    o = self.sigmoid(s)
                    self.value[i] = o


    def feedForward(self, Layer1, Layer2):
        a = 1

    def classification(self, X):
        return 0

    def netInput(self, X, w):
        return np.dot(X, w[1:]) + w[0]

    def sigmoid(self, _netInput):
        _netInput *= -1.0
        return 1.0 / (1.0 + np.exp(_netInput))

    def sigmoid_derivative(self, sigVal):
        sigVal *= -1
        return sigVal*(1 - sigVal)

"""
print(X_train.shape)
print(X_train.shape[0])
print(X_train.shape[1])
"""

NN = NeuralNetwork(3, 1, 0.1)
NN.fitData(X_train, y_train)
