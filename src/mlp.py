import math
import numpy as np
from random import uniform
from data import get_vectors



class mlp:
    def __init__(self, n_hidden_nodes, n_outputs):
        data = get_vectors()["training"]
        n_inputs = len(data[0]) - 2
        self.__n_inputs = n_inputs
        self.__n_outputs = n_outputs
        self.__n_hidden_nodes = n_hidden_nodes
        # initial weights 
        self.hidden_layer_weight = [[uniform(-1.0, 1.0)  for i in range(n_inputs)] for i in range(n_hidden_nodes)]
        self.output_layer_weight = [[uniform(-1.0, 1.0)  for i in range(n_hidden_nodes)] for i in range(n_outputs)]
        
        self.output_neurons = self.train(get_vectors()["training"])
        # number of epochs
        self.n_epochs = 0

    def train(self, dataset):

        for example in dataset:
            
            # self._forward_prop(self.hidden_layer, self.output_layer, attribute_vector)
            # self._backprop(self.hidden_layer, self.output_layer, output_vector, dataset[:][len(example)-1], hidden_vector, attribute_vector)
            
            # target vector for each example follows 80% 20% 
            # attribute_vector = dataset[:][len(example)-1]

            attribute_vector = example[1:len(example)-1]
            target_vector = self.__getTarget_vec(example) 

            hidden_layer_neurons, output_layer_neurons = self._forward_prop(self.hidden_layer_weight, self.output_layer_weight, attribute_vector)
            self.hidden_layer_weight, self.output_layer_weight = self._backprop(self.hidden_layer_weight, self.output_layer_weight, 
                                                                output_layer_neurons,target_vector , hidden_layer_neurons, attribute_vector)

        return hidden_layer_weight

    def get_classification(self):
        return self.__get_classification(self.output_neurons)

    def print_weights(self):
        #or output to file?? idk what to call this one
        pass
    

    # Return the label of each example 
    def __get_label(self, example):
        return example[len(example)-1]

    # Return target vector of each example vector using 80% 20% 
    def __getTarget_vec(self, example):
        target_vec = [0 for i in range(0,8)]
        example_label = self.__get_label(example)
        target_vec[example_label-1] = 0.8
        for el in target_vec: 
            if el == 0:
                el = 0.2

        return target_vec

    # Return (hidden_weight, output_weights) and (hidden_neurons, output_neurons)
    def _forward_prop(self, hidden_layer, output_layer, attribute_vector):
        hidden_layer_output = []
        output_layer_output= []
        
        # go through each hidden layer node
        for hidden_node in hidden_layer:
            # print("hidden node ", hidden_node)
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
            for i in range(len(hidden_layer)):
                swixi += hidden_layer_output[i] * output_node[i]

            # keep track of the output of the output layer
            output_layer_output.append(self.__sigmoid(swixi))

        return hidden_layer_output, output_layer_output

    def __sigmoid(self, swixi):
        return 1/(1 + math.pow(math.e, -swixi))

    
    def _backprop(self, hidden_layer_weight, output_layer_weight, output_layer_neurons, target_vector, hidden_layer_neurons, attribute_vector, eta=0.1):
        hidden_layer_weight = np.array(hidden_layer_weight)
        output_layer_weight = np.array(output_layer_weight)
        output_layer_neurons = np.array(output_layer_neurons)
        target_vector = np.array(target_vector)
        hidden_layer_neurons = np.array(hidden_layer_neurons)
        attribute_vector = np.array(attribute_vector)
        # num_hid = self.__n_hidden_nodes
        # num_out = self.__n_outputs

        num_hid = hidden_layer_neurons.size
        num_out = output_layer_neurons.size
        print('num hid' ,num_hid)
        print('num out ', num_out)
   
        
        output_responsibility = np.multiply(np.multiply(output_layer_neurons, (1 - output_layer_neurons)), (target_vector - output_layer_neurons))
        hidden_responsibility = np.multiply(np.multiply(hidden_layer_neurons, (1 - hidden_layer_neurons)), output_responsibility.dot(output_layer_weight))
        print('hidden_responsibility ', hidden_responsibility)

        output_layer_weight = output_layer_weight + eta*np.multiply(np.array([output_responsibility,]*num_hid).transpose(), hidden_layer_neurons)
        print('output_layer_weight ', output_layer_weight)


        # To Do: need to debug here!!!! size are not matched? 
        hidden_layer_weight = hidden_layer_weight + eta*np.multiply(np.array([hidden_responsibility,]*num_out).transpose(), attribute_vector)

        return hidden_layer_weight, output_layer_weight




    # def _backprop(self, hidden_layer, output_layer, output_vector, target_vector, hidden_vector, attribute_vector, eta=0.1):
    #     hidden_layer = np.array(hidden_layer)
    #     output_layer = np.array(output_layer)
    #     output_vector = np.array(output_vector)
    #     target_vector = np.array(target_vector)
    #     hidden_vector = np.array(hidden_vector)
    #     attribute_vector = np.array(attribute_vector)
    #     num_hid = self.__n_hidden_nodes
    #     num_out = self.__n_outputs

    #     print("shape ", hidden_vector.shape)
    #     print("hidden layer ", hidden_layer)
    #     print("output layer ", output_layer)

    #     output_responsibility = np.outer(np.outer(output_vector, (1 - output_vector)), (target_vector - output_vector))
    #     hidden_responsibility = np.outer(np.outer(hidden_vector, (1 - hidden_vector)), output_responsibility.dot(output_layer))

    #     output_layer = output_layer + eta*np.outer(np.array([output_responsibility,]*num_hid).transpose(), hidden_vector)
    #     hidden_layer = hidden_layer + eta*np.outer(np.array([hidden_responsibility,]*num_out).transpose(), attribute_vector)

    #     return hidden_layer, output_layer


    def __get_classification(self, output_neurons):
        # classifier chooses the class whose output neuron has return the highest value 
        return output_neurons.index(max(output_neurons)) + 1

if __name__ == "__main__":
    # Testing forward propagation
    # Example from book, table 5.1. slighty off due to rounding i think, but it shouldn't matter
    hidden_layer_weight = [
        [-1.0, 0.5],
        [0.1, 0.7]
    ]

    output_layer_weight = [
        [0.9, 0.5],
        [-0.3, -0.1]
    ]

    attribute_vector = [0.8, 0.1]

  
    # MLP = mlp(2,2)
    # forward_prop_results =  MLP._forward_prop(hidden_layer_weight, output_layer_weight, attribute_vector)
    # # Hidden node outputs
    # print(forward_prop_results[0])
    # # Output node outputs
    # print(forward_prop_results[1])
 
    
    # # Testing backpropagation
    # # Example from Table 5.3 in Kubat
    # hidden_layer = [ [-1.0, 1.0],
    #                  [1.0, 1.0] ]

    # output_layer = [ [1.0, 1.0],
    #                  [-1.0, 1.0] ]

    # output_vector = [0.65, 0.59]
    # target_vector = [1.0, 0.0]
    # hidden_vector = [0.12, 0.5]
    # attribute_vector = [1.0, -1.0]

    # backprop_results = MLP._backprop(hidden_layer, output_layer, output_vector, target_vector, hidden_vector, attribute_vector)
    # # Hidden node outputs/weights
    # print("hidden nodes: " ,backprop_results[0])
    # # Output node outputs/weights
    # print("output nodes: ", backprop_results[1])

 
    # print(MLP.hidden_layer)
    # print(MLP.output_layer)
    # for i in MLP.hidden_layer: 
    #     print('----------')
    #     print(i)
    # for output in MLP.output_layer:
    #     print('***********')
    #     print(output)



    MLP = mlp(5, 8)