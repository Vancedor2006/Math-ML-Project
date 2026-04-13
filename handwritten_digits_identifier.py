# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:58:41 2026

@author: Vezin
"""
import network
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784,100,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
