"""
mlp.py - mlp class that encapsulates all the data and functions of the mlp

Dennis La - Dennis.La@student.csulb.edu
Melissa Hazlewood - Melissa.Hazlewood@student.csulb.edu
Rowan Herbert - Rowan.Herbert@student.csulb.edu
Sophanna Ek - Sophanna.Ek@student.csulb.edu
"""
import math
import numpy as np
from random import uniform
from data import get_vectors
import sys


class mlp:
    def __init__(self, n_hidden_nodes, n_outputs, data):
        self.__n_inputs = len(data[0]) - 2
        # self.__n_inputs = n_inputs
        self.__n_outputs = n_outputs
        self.__n_hidden_nodes = n_hidden_nodes
        # initial weights 
        self.hidden_layer_weight = [[uniform(-1.0, 1.0)  for i in range(self.__n_inputs)] for i in range(self.__n_hidden_nodes)]
        self.output_layer_weight = [[uniform(-1.0, 1.0)  for i in range(self.__n_hidden_nodes)] for i in range(self.__n_outputs)]
        self.initial_weights = [self.hidden_layer_weight, self.output_layer_weight]

        # number of epochs
        self.n_epochs = 0
        # multipercentron weights after training
        self.training_weights = self.__train(data)



    # return the classification of each test example
    def get_classification(self, example):
        return self.__get_classification(example)

   # prints the initial and final weights of the hidden and output nodes
    def print_weights(self):

        print("--------Initial hidden layer weight-----------")
        for i in range(len(self.initial_weights[0])):
            print("Initial weights of hidden node", i, ":\n", self.initial_weights[0][i], "\n")
        print("\n--------Initial output layer weight-----------")
        for i in range(len(self.initial_weights[1])):
            print("Initial weights of output node", i, ":\n", self.initial_weights[1][i], "\n")

        print("\n-------Final hidden layer weight------------")
        for i in range(len(self.hidden_layer_weight)):
            print("Final weights of hidden node", i, ":\n", self.hidden_layer_weight[i], "\n")
        print("\n-------Final output layer weight -----------")
        for i in range(len(self.output_layer_weight)):
            print("Final weights of output node", i, ":\n", self.output_layer_weight[i], "\n")

    # prints the number of epochs
    def print_epochs(self):
        print("-------Epochs------------")
        print(self.n_epochs)

    # calculates precision, recall, sensitivity and specificity for a given class
    def print_rates_for_class(self, label, data):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        for example in data:
            # When an example is that given label, and the classifier says it is as well
            if label == self.__get_label(example) and self.get_classification(example) == label:
                true_pos += 1
            # When an example is that given label, but the classifier says it is not that label
            elif label == self.__get_label(example) and self.get_classification(example) != label:
                false_neg += 1
            # When an example is not that given label, but the classifier says it is that label
            elif label != self.__get_label(example) and self.get_classification(example) == label:
                false_pos += 1
            # When an example is not that given label, and the classifiers agrees that it is not that label as well
            elif label != self.__get_label(example) and self.get_classification(example) != label:
                true_neg += 1

        sensitivity = 0
        specificity = 0

        print("-----Rates for class:", label, "-----")
        # print precision
        if true_pos + false_pos > 0:
            print("Precision:", true_pos / (true_pos + false_pos), "\n")
        else:
            print("Precision:", 0, "\n")
        # print recall
        if true_pos + false_neg > 0:
            print("Recall:", true_pos / (true_pos + false_neg), "\n")
        else:
            print("Recall:", 0, "\n")
        # print sensitivity
        if true_pos + false_neg > 0:
            sensitivity = true_pos / (true_pos + false_neg)
            print("Sensitivity:", sensitivity, "\n")
        else:
            print("Sensitivity:", 0, "\n")
        # print specificity
        if true_neg + false_pos > 0:
            specificity = true_neg / (true_neg + false_pos)
            print("Specificity:", specificity, "\n")
        else:
            print("Specificity:", 0, "\n")

        print("Gmean:", math.sqrt(specificity * sensitivity), "\n")



    # return the multiperceptron weights
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
    
            # works as expected
            if(self.n_epochs == 800):
                break
            self.__train(dataset)
           
        
        return self.hidden_layer_weight, self.output_layer_weight

    
    # return true when all weights get ~0 
    def __is_change_negligible(self, old_weights, new_weights): 
        difference = abs(old_weights - new_weights)
        for row in difference:
            for el in row: 
                if el > sys.float_info.epsilon :
                    return False
        return True



    # Return the label of each example
    def __get_label(self, example):
        return example[len(example)-1]

    # Return target vector of each example vector using 80% 20%
    def __getTarget_vec(self, example):
        target_vec = [0 for i in range(0, 8)]
        example_label = self.__get_label(example)
        target_vec[example_label] = 0.8
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

    # logistic function
    def __sigmoid(self, swixi):
        return 1/(1 + math.pow(math.e, -swixi))


    #performs the backpropagation
    def __backprop(self, hidden_layer_weight, output_layer_weight, output_layer_neurons, target_vector, hidden_layer_neurons, attribute_vector, eta=0.1):
        hidden_layer_weight = np.array(hidden_layer_weight, dtype=np.float)
        output_layer_weight = np.array(output_layer_weight, dtype=np.float)
        output_layer_neurons = np.array(output_layer_neurons, dtype=np.float)
        target_vector = np.array(target_vector, dtype=np.float)
        hidden_layer_neurons = np.array(hidden_layer_neurons, dtype=np.float)
        attribute_vector = np.array(attribute_vector, dtype=np.float)
    
        num_hid = hidden_layer_neurons.size
        num_in = self.__n_inputs 
        
        output_responsibility = np.multiply(np.multiply(output_layer_neurons, (1 - output_layer_neurons)), (target_vector - output_layer_neurons))
        hidden_responsibility = np.multiply(np.multiply(hidden_layer_neurons, (1 - hidden_layer_neurons)), output_responsibility.dot(output_layer_weight))

        output_layer_weight = output_layer_weight + eta*np.multiply(np.array([output_responsibility,]*num_hid).transpose(), hidden_layer_neurons)

        hidden_layer_weight = hidden_layer_weight + eta*np.multiply(np.array([hidden_responsibility,]*num_in).transpose(), attribute_vector)

        return hidden_layer_weight, output_layer_weight

    
    # return classification for the example
    def __get_classification(self,example):
        
        # classifier chooses the class whose output neuron has return the highest value 
        [hidden_neurons, output_neurons] = self.__forward_prop(self.training_weights[0], self.training_weights[1], example[1:len(example)-2])

        return output_neurons.index(max(output_neurons))

    # return the accuracy of the 
    def get_accuracy(self, dataset):
        num_correct = 0
        for example in dataset:
            if self.get_classification(example) == self.__get_label(example):
                num_correct += 1
        return num_correct / len(dataset)

