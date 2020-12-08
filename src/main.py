from mlp import mlp
from data import get_vectors




def main():
    print("***EXECUTION INITIATED***")
    
    MLP = mlp(8, 8)
    
 
    data = get_vectors()['holdout']

    for example in data: 
        MLP.get_classification(example)



if __name__ == "__main__":
    main()