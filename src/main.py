"""
main.py - The entry point of the project. Prints out all the output
for the project

Dennis La - Dennis.La@student.csulb.edu
Melissa Hazlewood - Melissa.Hazlewood@student.csulb.edu
Rowan Herbert - Rowan.Herbert@student.csulb.edu
Sophanna Ek - Sophanna.Ek@student.csulb.edu
"""
from mlp import mlp
from data import get_vectors, trainingSet, training


def main():
    print("***EXECUTION INITIATED***")
    # train the data
    training_data = get_vectors()['training1and2']
    # create MLP with 12 hidden nodes and 8 output nodes
    MLP = mlp(12, 8, training_data)
    # test the holdout set
    holdout_data = get_vectors()['holdout']
    for example in holdout_data:
        MLP.get_classification(example)

    print()
    # printing the weights and the epochs
    MLP.print_weights()
    MLP.print_epochs()
        
    #Validation
    print("----------Validation----------")
    validation_set = get_vectors()['training2']
    print("\nConfusion Matrix Rates on Holdout:")
    MLP.print_rates_for_class(3, holdout_data)
    MLP.print_rates_for_class(2, holdout_data)
    print("\nConfusion Matrix Rates on Validation Set:")
    MLP.print_rates_for_class(3, validation_set)
    MLP.print_rates_for_class(2, validation_set)
    accuracy_rate = MLP.get_accuracy(holdout_data)
    error_rate = 1 - accuracy_rate
    print("MLP Accuracy and Error Rates for Holdout:")
    print("Accuracy rate =", accuracy_rate)
    print("Error rate =", error_rate)
    accuracy_rate = MLP.get_accuracy(validation_set)
    error_rate = 1 - accuracy_rate
    print("\nMLP Accuracy and Error Rates for Validation:")
    print("Accuracy rate =", accuracy_rate)
    print("Error rate =", error_rate)


if __name__ == "__main__":
    main()