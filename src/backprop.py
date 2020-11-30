import numpy as np

def backpropagation(hidden_layer, output_layer, output_vector, target_vector, hidden_vector, attribute_vector, eta=0.1):
	hidden_layer = np.array(hidden_layer)
	output_layer = np.array(output_layer)
	output_vector = np.array(output_vector)
	target_vector = np.array(target_vector)
	hidden_vector = np.array(hidden_vector)
	attribute_vector = np.array(attribute_vector)
	num_hid = len(hidden_vector)
	num_out = len(output_vector)

	output_responsibility = np.multiply(np.multiply(output_vector, (1 - output_vector)), (target_vector - output_vector))
	hidden_responsibility = np.multiply(np.multiply(hidden_vector, (1 - hidden_vector)), output_responsibility.dot(output_layer))

	output_layer = output_layer + eta*np.multiply(np.array([output_responsibility,]*num_hid).transpose(), hidden_vector)
	hidden_layer = hidden_layer + eta*np.multiply(np.array([hidden_responsibility,]*num_out).transpose(), attribute_vector)
	print(np.array([hidden_responsibility,]*num_out).transpose())

	return hidden_layer, output_layer

if __name__ == "__main__":
	# Example from Table 5.3 in Kubat
	hidden_layer = [ [-1.0, 1.0],
					 [1.0, 1.0] ]

	output_layer = [ [1.0, 1.0],
	  				 [-1.0, 1.0] ]

	output_vector = [0.65, 0.59]
	target_vector = [1.0, 0.0]
	hidden_vector = [0.12, 0.5]
	attribute_vector = [1.0, -1.0]

	backprop_results = backpropagation(hidden_layer, output_layer, output_vector, target_vector, hidden_vector, attribute_vector)
	# Hidden node outputs/weights
	print(backprop_results[0])
	# Output node outputs/weights
	print(backprop_results[1])
