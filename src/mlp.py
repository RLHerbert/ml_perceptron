import math
import numpy as np
from random import uniform



class mlp:
    def __init__(self, n_inputs, n_hidden_nodes, n_outputs):
        self.__n_inputs = n_inputs
        # initial weights 
        self.hidden_layer = [{'weights':[uniform(-1.0, 1.0)  for i in range(n_inputs + 1)]} for i in range(n_hidden_nodes)]
        self.output_layer = [{'weights':[uniform(-1.0, 1.0)  for i in range(n_hidden_nodes + 1)]} for i in range(n_outputs)]
        self.output_neurons = self.train()

    def train(self):
        pass

    def get_classification(self):
        return self.__get_classification(output_neurons)

    def print_weights(self):
        #or output to file?? idk what to call this one
        pass

    def _forward_prop(self,hidden_layer, output_layer, attribute_vector):
        hidden_layer_output = []
        output_layer_output = []

        # go through each hidden layer node
        for hidden_node in hidden_layer:
            swixi = 0
            # go through each attribute
            for i in range(len(attribute_vector)):
                swixi += attribute_vector[i] * hidden_node[i]

            # keep track of the output of the hidden layer
            hidden_layer_output.append(self.__sigmoid(swixi))

        # go through each output layer node
        for output_node in output_layer:
            swixi = 0
            # go through each hidden node's output
            for i in range(len(hidden_layer_output)):
                swixi += hidden_layer_output[i] * output_node[i]

            # keep track of the output of the output layer
            output_layer_output.append(self.__sigmoid(swixi))

        return hidden_layer_output, output_layer_output

    def __sigmoid(self, swixi):
        return 1/(1 + math.pow(math.e, -swixi))

    def _backprop(self, hidden_layer, output_layer, output_vector, target_vector, hidden_vector, attribute_vector, eta=0.1):
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

        return hidden_layer, output_layer


    def __get_classification(self, output_neurons):
        # classifier chooses the class whose output neuron has return the highest value 
        return output_neurons.index(max(output_neurons)) + 1

if __name__ == "__main__":
    # Testing forward propagation
    # Example from book, table 5.1. slighty off due to rounding i think, but it shouldn't matter
    hidden_layer = [
        [-1.0, 0.5],
        [0.1, 0.7]
    ]

    output_layer = [
        [0.9, 0.5],
        [-0.3, -0.1]
    ]

    attribute_vector = [0.8, 0.1]

    MLP = mlp(2,2,2)

    forward_prop_results =  MLP._forward_prop(hidden_layer, output_layer, attribute_vector)
    # Hidden node outputs
    print(forward_prop_results[0])
    # Output node outputs
    print(forward_prop_results[1])
    
    # Testing backpropagation
    # Example from Table 5.3 in Kubat
    hidden_layer = [ [-1.0, 1.0],
                     [1.0, 1.0] ]

    output_layer = [ [1.0, 1.0],
                     [-1.0, 1.0] ]

    output_vector = [0.65, 0.59]
    target_vector = [1.0, 0.0]
    hidden_vector = [0.12, 0.5]
    attribute_vector = [1.0, -1.0]

    backprop_results = MLP._backprop(hidden_layer, output_layer, output_vector, target_vector, hidden_vector, attribute_vector)
    # Hidden node outputs/weights
    print("hidden nodes: " ,backprop_results[0])
    # Output node outputs/weights
    print("output nodes: ", backprop_results[1])

 
    print(MLP.hidden_layer)
    print(MLP.output_layer)