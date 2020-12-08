"""
This file is just for testing stuff before putting it into mlp.py
"""


def print_weights(init_hidden_weights, init_ouput_weights,
                  final_hidden_weights, final_ouput_weights):
    print("Initial hidden weights:")
    for i in range(len(init_hidden_weights)):
        print("Initial weights of hidden node", i, ":", init_hidden_weights[i])
    print()
    print("Initial output weights:")
    for i in range(len(init_ouput_weights)):
        print("Initial weights of output node", i, ":", init_ouput_weights[i])
    print()
    print("Final hidden weights:")
    for i in range(len(final_hidden_weights)):
        print("Final weights of hidden node", i, ":", final_hidden_weights[i])
    print()
    print("Final output weights:")
    for i in range(len(final_ouput_weights)):
        print("Final weights of output node", i, ":", final_ouput_weights[i])


def print_epochs():
    #print(self.epochs)
    pass


def get_accuracy(mlp, test_set):
    pass


if __name__ == "__main__":
    hidden_layer = [
        [-1.0, 0.5],
        [0.1, 0.7]
    ]

    output_layer = [
        [0.9, 0.5],
        [-0.3, -0.1]
    ]

    final_hidden_layer = [
        [-.9, 0.4],
        [0.6, 0.5]
    ]

    final_output_layer = [
        [0.8, 0.2],
        [-0.5, -0.9]
    ]

    print_weights(hidden_layer, output_layer, final_hidden_layer, final_output_layer)
