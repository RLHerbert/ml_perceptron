import math


def __sigmoid(swixi):
    return 1/(1 + math.pow(math.e, -swixi))


def __foward_prop(hidden_layer, output_layer, attribute_vector):
    hidden_layer_output = []
    output_layer_output = []

    # go through each hidden layer node
    for hidden_node in hidden_layer:
        swixi = 0
        # go through each attribute
        for i in range(len(attribute_vector)):
            swixi += attribute_vector[i] * hidden_node[i]

        # keep track of the output of the hidden layer
        hidden_layer_output.append(__sigmoid(swixi))

    # go through each ouput layer node
    for output_node in output_layer:
        swixi = 0
        # go through each hidden node's output
        for i in range(len(hidden_layer_output)):
            swixi += hidden_layer_output[i] * output_node[i]

        # keep track of the output of the output layer
        output_layer_output.append(__sigmoid(swixi))

    return hidden_layer_output, output_layer_output


if __name__ == "__main__":

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

    forward_prop_results = __foward_prop(hidden_layer, output_layer, attribute_vector)
    # hidden node outputs
    print(forward_prop_results[0])
    # output node outputs
    print(forward_prop_results[1])
