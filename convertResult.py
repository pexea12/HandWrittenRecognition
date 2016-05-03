import numpy as np

def convertArrayToNumber(arr):
	"""Convert array into number:
			[0 1 0 0 0 0 0 0 0 0] -> 2
			[0 0 0 1 0 0 0 0 0 0] -> 4
			[0 0 0 0 0 0 0 0 0 1] -> 0
	"""
	y = np.zeros((arr.shape[0], 1))
	for i in range(arr.shape[0]):
		for j in range(10):
			if arr[i, j] == 1:
				y[i, 0] = j + 1
	return y
	
def convertNumberToArray(y):
	# Convert y into (10 x 1) matrix
	
	# For example: 4 is (0 0 0 1 0 0 0 0 0 0) (4th member is 1)
	#			   1 is (1 0 0 0 0 0 0 0 0 0) (1th member is 1)
	#			   0 is (1 0 0 0 0 0 0 0 0 0) (10th member is 1)
	
	newY = np.zeros((y.shape[0], 10), dtype=np.int)
	for i in range(y.shape[0]):
		index = (y[i, 0] - 1) % 10
		newY[i, index] = 1
		
	return newY