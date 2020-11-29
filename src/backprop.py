import numpy as np

ETA = 0.1 #learning rate

def sigmoid(x):
	return 1.0/(1 - np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(X) * (1 - sigmoid(x))


def backpropagation(hidden_layer, output_layer, attribute_vector):
	# TODO: WRONG... FIX
	hidden_layer = np.array(hidden_layer)
	output_layer = np.array(output_layer)
	attribute_vector = np.array(attribute_vector)

	for output_node in output_layer:
		output_responsibility = np.multiply(np.multiply(output_layer, (1 - output_layer)), attribute_vector)
	for hidden_node in hidden_layer:
		hidden_responsibility = np.multiply(np.multiply(hidden_layer, (1 - hidden_layer)), output_responsibility.dot(output_layer))

	output_layer = output_layer + ETA*np.multiply(output_responsibility, hidden_layer)
	hidden_layer = hidden_layer + ETA*np.multiply(hidden_responsibility, attribute_vector)

	return hidden_layer, output_layer

if __name__ == "__main__":
	# TODO: WRONG... check example
	# example copied from dennis lol
    # example from book, table 5.1. slighty off due to rounding i think, but it shouldn't matter
    hidden_layer = [
        [-1.0, 0.5],
        [0.1, 0.7]
    ]

    output_layer = [
        [0.9, 0.5],
        [-0.3, -0.1]
    ]

    attribute_vector = [0.8, 0.1]

    forward_prop_results = backpropagation(hidden_layer, output_layer, attribute_vector)
    # hidden node outputs
    print(forward_prop_results[0])
    # output node outputs
    print(forward_prop_results[1])