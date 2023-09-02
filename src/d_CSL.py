#!/usr/bin/env python3.11
# coding: utf-8

import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit, njit
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class GcStar:
    """
    Implementation of Granger causality from causal Bayesian network perspective. 
    Methods:
        fit(self, data: np.ndarray) 
        get_connectivity_matrix(self)
        
    """

    corr_: Optional[np.ndarray]
    pVal_corr_: Optional[np.ndarray]
    inv_corr_: Optional[np.ndarray]
    pVal_inv_corr_: Optional[np.ndarray]

    def __init__(self, n_perm: int, n_pasts: int, n_lags: int, temporal: bool = True):
        """

        Args:
            n_perm: number of permutations (default = 1000)
            n_pasts: number of past states
            n_lags: maximum allowable lags 
            temporal: defines if data is time series or iid
        """
        self.n_perm = n_perm
        self.n_pasts = n_pasts
        self.n_lags = n_lags
        self.temporal = temporal

        self.data = None
        self.shifted_data = None

        self.corr_ = None
        self.pVal_corr_ = None
        self.inv_corr_ = None
        self.pVal_inv_corr_ = None

    def is_time_series(self) -> bool:
        return self.temporal

    def get_number_of_lags(self) -> int:
        """
        Collects an integer valued maximum lag desired for use in the analysis
        """
        return self.n_lags

    @staticmethod
    def __shift_data(arr: np.ndarray, n_past: int) -> np.ndarray:
        """
        Creates shifted versions of original data based on the number of pasts required.
        Concatenate the shifted arrays and return the new data.

        Examples: Given original data as X,
            trimmed_arr = __shifted_data(X, n_past = n)
            with trimmed_arr = {X_{t}, X_{t-1}, \dots, X_{t-n}}^T
        Args:
            arr: original data recorded or obtained from experiments
            n_past: number of pasts defining the number of shifts 

        Returns:
            Shifted data. 
        """

        if n_past == 0:
            return arr

        trimmed_arr = arr[:, n_past:]
        for i in range(n_past):
            idx1 = n_past - 1 - i
            idx2 = -i - 1

            trimmed_arr = np.r_[trimmed_arr, arr[:, idx1:idx2]]

        return trimmed_arr


    @jit(nopython=True)
    def __perm_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes p_value of correlation of two iid generated variables.
        Args:
            x: (array like, vector): Realisation of a variable x
            y: (array like, vector): Realisation of a variable y
        Returns:
            p_value
        """
        count = 0
        corr_1 = np.corrcoef(x, y)[1, 0]

        for j in range(self.n_perm):
            x_copy = np.copy(x)
            if not self.temporal:
                np.random.shuffle(x_copy)
                x_ = x_copy
            else:
                x_ = np.roll(x_copy, np.random.randint(30, len(x)), 1)

            corr_2 = np.corrcoef(x_, y)[1, 0]
            if np.abs(corr_2) >= np.abs(corr_1):
                count += 1
        return count / self.n_perm

    def __get_number_of_neurons(self, trimmed_arr):
        """
        Computes the number of neurons/variables from the shape of the shifted data  
        Args:
            trimmed_arr: shifted data from self.__shift_data()
        Returns:
            number of variables/neurons in data
        """
        return trimmed_arr.shape[0] / (self.n_pasts + 1)

    def __correlation_func(self, trimmed_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        # name compute_dependence_with_corelation()
        Computes unconditional dependence of any variable pair using Pearson's correlation as dependence metric
        Args:
            trimmed_arr: shifted data from __shift_data()
        Returns:
            corr[:, :n]: np.ndarray shape [trimmed_arr.shape[0], self.n_neur]
                Correlation matrix of the data but done on the shifted data, however, we select the portion
                that is most relevant for us by slicing the matrix into shape required

            pVal_corr: np.ndarray with shape as the correlation matrix.
                contains the p-values of individual elements in the correlation matrix
        """
        corr = np.abs(np.corrcoef(trimmed_arr))
        n = trimmed_arr.shape[0]
        n_neur = self.__get_number_of_neurons(trimmed_arr)
        pVal_corr = np.zeros((n, n_neur))

        for i in range(n):
            for j in range(n_neur):
                pVal_corr[i, j] = self.__perm_test(trimmed_arr[i, :], trimmed_arr[j, :])
        return corr[:, :n], pVal_corr

    def __residual(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Computes the residuals of a variable x by regressing a conditioning set z out of it
        Args:
            x: Variable in question to regress out the conditioning set
            z: Conditioning set based on whether c-GC or fc-GC was used.
        return:
            the residual of x conditioned on z
        """

        model = LinearRegression(fit_intercept=True)
        model.fit(z.T, x)

        coefs = model.coef_
        intercept = model.intercept_

        return x - np.dot(coefs, z) - intercept

    def __conditioning_set(self, i, j):
        """
        Identifies the variables in the conditioning set for granger causality implementation.
        Args:
            i: (int) number of permutations for p_value computations TODO: CHECK IF I AN J ARE CORRECTLY ASSIGNED
            j: (int) number of pasts desired
        Returns: correlation, corresponding p-Values, inverse correlation and the p-valeus
        """
        n = self.data.shape[0]
        k = i // n

        z_ = np.delete(self.shifted_data, np.r_[np.arange(k * n), [i]], axis=0)
        y_ = self.__shift_data(self.data[j, :].reshape((1, self.data.shape[1])), self.n_pasts)
        z = np.r_[z_, y_[1:k]]

        return z

    def __inv_correlation_func(self, trimmed_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes conditional dependency of any variable pair with Pearson's correlation as the dependence metric.
        It computes the residual of any pair in question with the conditioning set of choice (c-GC or fc-GC style).
        Args:
            trimmed_arr: This is the modified data obtained from self.__shift_data()
        Returns:
            inv_corr: np.ndarray shape [trimmed_arr.shape[0], self.n_neur]
                Correlation matrix of the data but done on the shifted data, however, we select the portion
                that is most relevant for us by slicing the matrix into shape required

            pVal_inv_corr: np.ndarray with shape as the correlation matrix
                contains the p-values of individual elements in the correlation matrix
        """
        n = trimmed_arr.shape[0]
        n_neur = self.__get_number_of_neurons(trimmed_arr)

        inv_corr = np.zeros((n, n_neur))
        pVal_inv_corr = np.zeros((n, n_neur))

        for i in range(0, n):
            for j in range(0, n_neur):
                x = trimmed_arr[i]
                y = trimmed_arr[j]
                z = self.__conditioning_set(i, j)

                x_res = self.__residual(x, z)
                y_res = self.__residual(y, z)

                inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                pVal_inv_corr[i, j] = self.__perm_test(x_res, y_res)
        return inv_corr, pVal_inv_corr

    def fit(self, data: np.ndarray):
        """
        Fits c-gc to data 
        Args: 
            data: array-like of shape (n_neur, T)
                where `n_neur` is the number of neurons or variables
                and `T` is the time or number of samples 

        Returns:
            self : object
                Returns the instance itself
        """
        self.data = data.copy()
        self.shifted_data = self.__shift_data(self.data, self.n_pasts)
        self.corr_, self.pVal_corr_ = self.__correlation_func(self.shifted_data)  # TODO: remove parameter and just access class variable
        self.inv_corr_, self.pVal_inv_corr_ = self.__inv_correlation_func(self.shifted_data)

    def get_connectivity_matrix(self, alpha: float = 0.05, beta: float = 0.001) -> np.ndarray:
        """
        Computes the connectivity matrix from significant conditional and unconditional links
            based on the maximum lag allowed for the analysis. 
        Args:
            alpha: float, Significance level for unconditional dependence (default value 0.05) 
            beta: float, Significance level for conditional dependence (default value 0.001)

        Returns:
            Connectivity matrix of shape (n_neur, n_neur)
        """
        # compute significant correlation and inverse correlation matrices
        sig_corr = np.multiply(self.corr_, self.pVal_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.pVal_inv_corr_ <= beta)

        # extended connectivity matrix
        inferred = np.logical_and(sig_corr, sig_inv)

        b = 0
        all_ = []
        n_neur = inferred.shape[1]

        # merging results for the GC order used
        for a in range(self.n_pasts + 1):
            all_.append(inferred[a * n_neur:(a + 1) * n_neur, b * n_neur:(b + 1) * n_neur])

        new_inf = all_[0]

        for i in range(1, self.n_lags):  # use 'nn' if all matrices are to be used, here i take out the last one
            new_inf = np.logical_or(new_inf, all_[i])

        # multiplied with correlation to return connections strength
        new_inf = np.multiply(self.corr_[:n_neur, :], new_inf)
        # remove self connections (diagonal connections)
        np.fill_diagonal(new_inf, 0)

        return new_inf