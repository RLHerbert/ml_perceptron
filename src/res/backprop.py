import numpy as np

def sigmoid(x):
	return 1.0/(1 - np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(X) * (1 - sigmoid(x))


def backpropagation()