# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:49:45 2026

@author: Vezin
"""

"""
mnist_loader
~~~
A library to load the MNIST image data.  
"""

import pickle
import gzip
import numpy as np

def load_data():
    
    f = gzip.open('C:/Users/Vezin/--Python--School/MP&S_Group_Project/neural-networks-and-deep-learning-master/data/mnist.pkl.gz','rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    # Format the training data
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    # Format the validation data
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    # Format the test data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e