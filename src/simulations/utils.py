#!/usr/bin/env python3
# coding: utf-8

# imports 
import numpy as np
import matplotlib.pyplot as plt
from d_CSL import *
from dataload import *

# genderate a random ground truth conneectivity matrix A

A = adj_mtx(n_neur)

# simulate data 
noise = continuous_noise_fun(n_neur, l)
X = simulate_data()

# genderate a random ground truth conneectivity matrix A

A = adj_mtx(n_neur)

# simulate data 
noise = continuous_noise_fun(n_neur, l)
X = simulate_data()


def Simulations():
    def __init__(self, data, n_perm, n_pasts, temporal):
        self.number_of_perm = n_perm
        self.number_of_lags = n_pasts
        self.traces = data
        self.temporal = True

    def get_number_of_variables(self):
        return self.number_of_variables

    

    def simulate_data(self, len_of_variables):
        self.simulate_data()

        
    def mat_func(self, A, inf):
        """
        For plotting results with different colors for metrics in the connfusion matrix 
        :param A:
        :param inf:
        :return:
        """
        return 40*np.logical_and(A!=0,inf!=0) + 30*np.logical_and(A==0,inf!=0) + 20*np.logical_and(A!=0,inf==0)+10*np.logical_and(A==0,inf==0)
