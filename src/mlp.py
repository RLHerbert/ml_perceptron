import math
import numpy as np
from random import uniform
from data import get_vectors
import sys



class mlp:
    def __init__(self, n_hidden_nodes, n_outputs):
        data = get_vectors()["training"]
        self.__n_inputs = len(data[0]) - 2
        self.__n_outputs = n_outputs
        self.__n_hidden_nodes = n_hidden_nodes
        # initial weights 
        self.hidden_layer_weight = [[uniform(-1.0, 1.0)  for i in range(self.__n_inputs)] for i in range(self.__n_hidden_nodes)]
        self.output_layer_weight = [[uniform(-1.0, 1.0)  for i in range(self.__n_hidden_nodes)] for i in range(self.__n_outputs)]
        # number of epochs
        self.n_epochs = 0
        # multipercentron weights after training
        self.training_weights = self.__train(get_vectors()['training'])
               


    # return the classification of each test example
    def get_classification(self, example):
        return self.__get_classification(example)

    # TODO
    def print_weights(self):
        #or output to file?? idk what to call this one
        pass


    
    def __train(self, dataset):
        for example in dataset:       
            attribute_vector = example[1:len(example)-1]
            # target vector for each example follows 80% 20% 
            target_vector = self.__getTarget_vec(example) 

            # perform forward propagation
            hidden_layer_neurons, output_layer_neurons = self.__forward_prop(self.hidden_layer_weight, self.output_layer_weight, attribute_vector)
            prev_hidden_layer_weight = self.hidden_layer_weight
            prev_output_layer_weight = self.output_layer_weight
            # perform backpropagation erorr
            self.hidden_layer_weight, self.output_layer_weight = self.__backprop(self.hidden_layer_weight, self.output_layer_weight, 
                       
                                                                output_layer_neurons,target_vector , hidden_layer_neurons, attribute_vector)
        self.n_epochs += 1

        while  not (self.__is_change_negligible(prev_hidden_layer_weight, self.hidden_layer_weight) and
                        self.__is_change_negligible(prev_output_layer_weight, self.output_layer_weight)):
            
            ##stuck in stagnation -- exceed the allowable running time 
            # if(self.n_epochs == 100):
            #     self.hidden_layer_weight = [[uniform(-1.0, 1.0)  for i in range(self.__n_inputs)] for i in range(self.__n_hidden_nodes)]
            #     self.output_layer_weight = [[uniform(-1.0, 1.0)  for i in range(self.__n_hidden_nodes)] for i in range(self.__n_outputs)]
    
            # works as expected
            if(self.n_epochs == 250):
                break
            self.__train(dataset)
           
        
        return self.hidden_layer_weight, self.output_layer_weight

    
    # return true when all weights get ~0 
    def __is_change_negligible(self, old_weights, new_weights): 
        difference = abs(old_weights - new_weights)
        # print("different....")
        # print(difference)
        for row in difference:
            for el in row: 
                
                if el > sys.float_info.epsilon :
                # if np.testing.assert_array_almost_equal(el, sys.float_info.epsilon, decimal=6, err_msg='', verbose=True):
                    return False
        return True



    # Return the label of each example

    def __get_label(self, example):
        return example[len(example)-1]

    # Return target vector of each example vector using 80% 20%
    def __getTarget_vec(self, example):
        target_vec = [0 for i in range(0, 8)]
        example_label = self.__get_label(example)
        target_vec[example_label-1] = 0.8
        for i in range(len(target_vec)):
            if target_vec[i] == 0:
                target_vec[i] = 0.2

        return target_vec

    # Return hidden and ouput neurons of multi_percentrons
    def __forward_prop(self, hidden_layer, output_layer, attribute_vector):
        hidden_layer_output = []
        output_layer_output = []

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

    
    def __backprop(self, hidden_layer_weight, output_layer_weight, output_layer_neurons, target_vector, hidden_layer_neurons, attribute_vector, eta=0.1):
        hidden_layer_weight = np.array(hidden_layer_weight, dtype=np.float)
        output_layer_weight = np.array(output_layer_weight, dtype=np.float)
        output_layer_neurons = np.array(output_layer_neurons,dtype=np.float)
        target_vector = np.array(target_vector,dtype=np.float)
        hidden_layer_neurons = np.array(hidden_layer_neurons,dtype=np.float)
        attribute_vector = np.array(attribute_vector,dtype=np.float)
    
        num_hid = hidden_layer_neurons.size
        num_in = self.__n_inputs 
        
        output_responsibility = np.multiply(np.multiply(output_layer_neurons, (1 - output_layer_neurons)), (target_vector - output_layer_neurons))
        hidden_responsibility = np.multiply(np.multiply(hidden_layer_neurons, (1 - hidden_layer_neurons)), output_responsibility.dot(output_layer_weight))
        # print('hidden_responsibility ', hidden_responsibility)

        output_layer_weight = output_layer_weight + eta*np.multiply(np.array([output_responsibility,]*num_hid).transpose(), hidden_layer_neurons)
        # print('output_layer_weight ', output_layer_weight)

        hidden_layer_weight = hidden_layer_weight + eta*np.multiply(np.array([hidden_responsibility,]*num_in).transpose(), attribute_vector)

        return hidden_layer_weight, output_layer_weight



    # def _mean_squared_error(self,output_vec, target_vec):
    #     sum = 0
    #     for i in range(len(target_vec)):
    #         sum += (target_vec[i] - output_vec[i])** 2
    #     return sum * 1/len(target_vec)


    def __get_classification(self,example):
        # classifier chooses the class whose output neuron has return the highest value 
        [hidden_neurons, output_neurons] = self.__forward_prop(self.hidden_layer_weight, self.output_layer_weight, example[1:len(example)-2])
        print('the classification for this example is ...',output_neurons.index(max(output_neurons)) + 1 )
        return output_neurons.index(max(output_neurons)) + 1

    # return the accuracy of the 
    def __get_accuracy(self, dataset):
        correct = 0
        for example in dataset:
            if example[len(dataset[0]) - 1] == self.get_classification(example):
                correct += 1

        return correct / len(dataset)


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

 
   
    # test mlp()
    data = [10, 63, 36, 74, 9, 16, 77, 92, 62, 54, 58, 3]
    MLP = mlp(8, 8)
    
    print(MLP.get_classification(data))


    
