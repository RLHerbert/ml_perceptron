from mlp import mlp
from data import get_vectors, trainingSet, training

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

MLP = mlp(2,2)
forward_prop_results =  MLP._forward_prop(hidden_layer_weight, output_layer_weight, attribute_vector)
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


# test mlp()
data = [100, 53, 69, 43, 86, 63, 0, 57, 12, 52, 44, 2]
MLP = mlp(2, 8)


print(MLP.get_classification(data))


