# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2

import sys
import time
import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i

def digitilize_line(line, target = True):
    condition = {"Fair": -1, "Good": 2, "Like New": 3}
    area = {"aaa": 1,"bbb": 2,"ccc": 3,"ddd": 4,"eee": 5,"fff": -6,"ggg": -7,"hhh": -8,"jjj": -9,"kkk": -10}
    units = line.split(",")
    Y = int(float(units[1])) if target == True else 0
    X = [float(units[2]) / 1000.0,float(units[3]) / 1000.0, float(area[units[4]]), float(condition[units[5]]), -float(units[6])]
    # X = [X[0],math.cos(X[0]),math.sqrt(X[0]),math.sin(X[0]), X[0] * X[1],
    #      X[1],math.cos(X[1]),math.sqrt(X[1]),math.sin(X[1]), X[1] * X[1],
    #      X[2],math.cos(X[2]),math.sqrt(X[2]),math.sin(X[2]), X[2] * X[2],
    #      X[3],math.cos(X[3]),math.sqrt(X[3]),math.sin(X[3]), X[3] * X[3],
    #      X[4],math.cos(X[4]),math.sqrt(X[4]),math.sin(X[4]), X[4] * X[4]
    #     ]
    return X, Y

def load_data(train_filename, input_dim = 5):
    augmentation_ratio = 3
    L = file_len(train_filename) * augmentation_ratio
    print(L," records in the training data")

    X = np.zeros(( L, input_dim), dtype = np.float16)
    Y = np.zeros( L,             dtype = np.int8   )

    lines = list(open(train_filename))

    j = 0
    for k in range(augmentation_ratio):
        for i in range(1,len(lines)):   # Skip the head
            outs = digitilize_line(lines[i])
            X[j] = outs[0]
            Y[j] = outs[1]
            j += 1

    for i in range(10):
        print(X[i],Y[i])

    # Prepend the column of 1s for bias
    N, M  = X.shape
    print('Features dimensions:',N,M)

    outs = train_test_split(X, Y, test_size=0.30, random_state=RANDOM_SEED)
    return outs

def load_test(filename, input_dim = 5):
    L = file_len(filename)
    print(L," records in the test data")

    X = np.zeros(( L, input_dim), dtype = np.float16)
    Products = np.zeros(L, dtype = np.int32)
    lines = list(open(filename))
    for i in range(1,len(lines)):   # Skip the head
        Products[i-1] = int(lines[i].split(",")[0])
        outs = digitilize_line(lines[i], False)
        X[i-1] = outs[0]

    return X, Products
