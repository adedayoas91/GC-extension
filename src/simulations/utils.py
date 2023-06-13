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


class Simulations():
    def __init__(self, n_neur, n_perm, n_pasts, temporal):
        self.number_of_var = n_neur
        self.number_of_perm = n_perm
        self.number_of_lags = n_pasts
        self.temporal = True

    def get_number_of_variables(self):
        return self.number_of_variables

    
        def adj_mtx(n_neur):
        """
        Creates a random ground truth connectivity matrix for simulating data.
        Args:
            n_neu: (int) number of variables expected in data.

        Returns: shifted data

        """
        A = np.random.choice([0,0.5,0.85], p=[0.9,0.03,0.07], size=(n_neur,n_neur)) ### A is not the adjacency matrix in the typical sense
        A = 0.5*(A)
        A[0:10,0:10] =np.zeros((10,10))
        for n, i in enumerate(A):
            A[n][n]=1
        for n, i in enumerate(A):
            A[n]=A[n]/np.sum(A[n])
        return A


    def simulate_data(A, m, iid = bool, latency = bool):
        """
        Function to create a iid dataset for analysis

        Parameters
        A (matrix) - Adjacency matrix (A lower triangular matrix - topological order of the variables)
        m (int) - desired length of the variable
        iid (bool) - Specify data to be simulated.

        Returns
        data (matrix) with shape [A.shape[0],m]

        Example of A =  np.array([[0,0,0,0,0],
                                [1,0,0,0,0],
                                [0,1,0,0,0],
                                [1,1,0,0,0],
                                [0,0,1,0,0]])

        """
        np.random.seed(10)
        if iid==True:
            X = np.zeros([A.shape[0],m]).T
            for i, row in enumerate(X):
                for n, var in enumerate(row):
                    X[i, n] = np.random.normal(0, 0.1) + np.dot(A[n], X[i])
        else:
            X = np.zeros([A.shape[0],m]).T
            X[0] = np.random.randn(A.shape[0])
            noise = continuous_noise_fun(A.shape[0], m)
            for i, row in enumerate(X[:-1]):
                if latency == False:
                    X[i+1] = A @ X[i] + np.random.normal(0,0.25,A.shape[0])
                else:
                    X[i+1] = A @ X[i] + np.random.normal(0,0.25,A.shape[0]) + noise[:,i]

        return X.T


    def continuous_noise_fun(num, l):
        """

        :param num:
        :param l:
        :return:
        """
        xx = np.linspace(0,500,l)
        noise = np.zeros((num,l))
        for i in range(num):
            a = 2*np.random.normal(0,0.25,size=6)
            c = 500*(np.random.random(size=6))
            s = 1+100*(np.random.random(size=6))
            yy = 0*xx
            for j in range(6):
                yy = yy + a[j]*np.exp(-(xx-c[j])**2/s[j])
            noise[i] = yy
        return noise


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
