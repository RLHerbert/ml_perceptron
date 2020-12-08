from mlp import mlp
from data import get_vectors




def main():
    print("***EXECUTION INITIATED***")
    # create MLP    
    MLP = mlp(5, 8)
    # test the holdout set
    data = get_vectors()['holdout']
    for example in data: 
        MLP.get_classification(example)
    print(MLP.get_accuracy(data))
    MLP.print_weights()
        
    # TO DO: Validation


if __name__ == "__main__":
    main()