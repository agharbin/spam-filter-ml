from pyspark.mllib.linalg import SparseVector
from math import floor
import pickle
import sys
import random

def partition(data):
    # Randomize the data array
    random.shuffle(data)

    # Calculate the sizes of the needed data sets
    # 80% training data, 10% cross-validation data, 10% testing data
    rows = len(data)
    num_cv = num_test = int(floor(0.10 * rows))
    num_train = rows - num_cv - num_test

    cv_data = data[0:num_cv]
    test_data = data[num_cv:num_cv+num_test]
    train_data = data[num_cv+num_test:]

    return (cv_data, test_data, train_data)

if __name__ == "__main__":
    data_file = open(sys.argv[1],"r")
    data = pickle.load(data_file)
    
    cv_data, test_data, train_data = partition(data)

    cv_file = open("cross_validation_data.txt","w")
    test_file = open("test_data.txt","w")
    train_file = open("training_data.txt","w")

    pickle.dump(cv_data,cv_file)
    pickle.dump(test_data,test_file)
    pickle.dump(train_data,train_file)

    cv_file.close()
    test_file.close()
    train_file.close()
