#!/usr/bin/env python3
# coding: utf-8

# imports 
import numpy as np
import matplotlib.pyplot as plt
from d_CSL import *

# select an integer `n_neur` and `l` to denote the number of variables and length of the variabales
n_neur = int(input())
l = int(input())

# genderate a random ground truth conneectivity matrix A

A = adj_mtx(n_neur)

# simulate data 
noise = continuous_noise_fun(n_neur, l)
X = simulate_data()


def Simulations():
    def __init__():
        self.

        
    def mat_func(self, A, inf):
        """
        For plotting results with different colors for metrics in the connfusion matrix 
        :param A:
        :param inf:
        :return:
        """
        return 40*np.logical_and(A!=0,inf!=0) + 30*np.logical_and(A==0,inf!=0) + 20*np.logical_and(A!=0,inf==0)+10*np.logical_and(A==0,inf==0)
