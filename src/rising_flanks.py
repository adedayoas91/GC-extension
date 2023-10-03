#!/usr/bin/env python3
# coding: utf-8

import os
from typing import Self
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from typing import Tuple, Optional
from sklearn.metrics import mean_squared_error
import mat73
from numba import jit, njit, vectorize
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import concurrent.futures
import logging
import multiprocessing


#####################################################
#####################################################
### Rising Flanks
#####################################################
#####################################################

class RisingFlanks:
    """
        Implementation of Granger causality from causal Bayesian network
        perspective using only the rising flanks of the calcium imaging data.
        Methods:
            fit(self, data: np.ndarray)
            get_connectivity_matrix(self)
            plot_conn_mat_on_topography(self)
        """

    corr_: Optional[np.ndarray]
    pVal_corr_: Optional[np.ndarray]
    inv_corr_: Optional[np.ndarray]
    pVal_inv_corr_: Optional[np.ndarray]


    def __init__(self, n_perm: int, n_pasts: int, n_lags: int,
                                        f_s: float, seg_len: int):
        """
        Args:
            n_perm: number of permutations (default = 1000)
            n_pasts: number of past states
            n_lags: maximum allowable lags
            temporal: defines if data is time series or iid
        """

        logging.basicConfig(level=logging.INFO)
        self.logger = None  # Initialize logger when needed

        self.n_neur = None
        self.n_perm = n_perm
        self.n_pasts = n_pasts
        self.n_lags = n_lags
        self.seg_len = seg_len
        self.fs = f_s

        self.data = None
        self.shifted_data = None
        self.topography = None

        self.corr_ = None
        self.pVal_corr_ = None
        self.inv_corr_ = None
        self.pVal_inv_corr_ = None

    def moving_avg(self, data, win_size):
        """

        Args:
            data:
            win_size:

        Returns:

        """
        self.data = data.copy()
        i, moving_avg = 0, []
        while i < (len(data) - win_size + 1):
            window = data[i:i + win_size]
            win_avg = round(sum(window) / win_size, 4)
            moving_avg.append(win_avg)
            i += 1
        return moving_avg


    def ideal_lp(self, f_c, M):
        """

        Args:
            f_c:
            M:

        Returns:

        """
        amp = np.ones(M)
        amp[int(f_c * M / self.fs):-int(f_c * M / self.fs)] = 0
        phase = np.zeros(M)
        H_f = amp * np.exp(1j * phase)
        h_n = np.fft.fftshift(np.real(np.fft.ifft(H_f)))
        return h_n


    def shift_data(self, arr: np.ndarray) -> np.ndarray:   # , nn = None
        """
        Creates shifted versions of original data based on the number of pasts required.
        Concatenate the shifted arrays and return the new data.

        Examples: Given original data as X,
            trimmed_arr = __shifted_data(X, n_past = n)
            with trimmed_arr = {X_{t}, X_{t-1}, \dots, X_{t-n}}^T
        Args:
            arr: original data recorded or obtained from experiments
            n_pasts: number of pasts defining the number of shifts

        Returns:
            Shifted data.
        """
        # if nn != None:
        #     self.n_pasts = nn

        self.data = arr.copy()
        self.n_neur = self.data.shape[0]
        if self.n_pasts == 0:
            return arr

        trimmed_arr = arr[:, self.n_pasts:]
        for i in range(self.n_pasts):
            idx1 = self.n_pasts - 1 - i
            idx2 = -i - 1

            trimmed_arr = np.r_[trimmed_arr, arr[:, idx1:idx2]]
        self.shifted_data = trimmed_arr
        return trimmed_arr



    ## @jit(nopython=True)
    def __perm_test_(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes p_value of correlation of two iid generated variables.
        Args:
        x: (array like, vector): Realisation of a variable x
        y: (array like, vector): Realisation of a variable y

        Returns:
            p_value
        """
        count, corr_1 = 0, np.corrcoef(x, y)[1, 0]
        val = np.random.randint(5, len(x) - 8, self.n_perm)  # change 4 back to 30 and len(x)-30

        for j in range(self.n_perm):
            x_copy = np.copy(x)
            x_ = np.roll(x_copy, val[j])
            corr_2 = np.corrcoef(x_, y)[1, 0]

            if np.abs(corr_2) >= np.abs(corr_1):
                count += 1

        return count / self.n_perm


    def get_past(self, X: np.ndarray) -> np.ndarray:
        """
            Creates shifted versions of original data based on the number of pasts required.
            Concatenate the shifted arrays and return the new data.
        Args:
            X: np.ndarray of shape (num_vars, num_timesteps)
            n_past: int, number of timelags to consider

        Returns:
            np.ndarray of shape (n_past+1, num_vars, num_timesteps-n_past) with lags
            increasing along the first axis.
        """
        assert len(X.shape) == 2, \
            "X must be a 2-dimensional array"
        if self.n_pasts == 0:
            return X.copy().reshape(1, *X.shape)
        past_matrices = []
        for j in range(self.n_pasts + 1):
            X_past_j = X[:, self.n_pasts - j:X.shape[1] - j]
            past_matrices.append(X_past_j)
        X_past = np.stack(past_matrices)
        return X_past



    def get_conditioning_set(self, i: int, j: int) -> np.ndarray:
        """
        Identifies the variables in the conditioning set for
        Granger causality implementation.
        All indices in range(0, X.shape[0]) that are not `i_ind` are
        considered to be indices of latent variables.

        Args:
            i: (int) index of the cause variable
            j: (int) index of the effect variable

        Returns:
            The conditioning set containing relevant variables of type np.ndarray
            with 2-dimension, where each row represents a variable
            conditioned variable, and each column includes the historical
            values of said variable.

        """
        X = self.data.copy()
        num_vars = self.n_neur


        j_ind = j
        i_ind = i % num_vars
        i_lag = i // num_vars


        X_past = self.get_past(X)
        # get the latent variable indices
        all_indices = np.arange(num_vars)
        ij_mask = np.isin(all_indices, [i_ind, j_ind])
        z_indices = all_indices[~ij_mask]  # everything that isn't i or j is z
        # `X_past` has shape (n_past, num_vars, X.shape[1]-n_past)

        # From the independent variable, we want to return everything before but
        # not including the "current" value at `i_lag`
        i_past = X_past[i_lag + 1:, [i_ind], :]
        # i_past shape (history up to i_lag, 1, X.shape[1]-n_past)

        # For the latent variable, we want to return everything up to and at the
        # same time as the independent variable
        z_past = X_past[i_lag:, z_indices, :]
        # z_past shape (history up to i_lag+1, X.shape[0]-2, X.shape[1]-n_past)

        # For the dependent variable, we return all times in the past but not the
        # current value
        j_past = X_past[1:, [j_ind], :]
        # j_past shape (history up to current time, 1, X.shape[1]-n_past)

        # reshape everything to be compatible shape
        i_past_reshaped = i_past.reshape(-1, i_past.shape[-1])
        j_past_reshaped = j_past.reshape(-1, j_past.shape[-1])
        z_past_reshaped = z_past.reshape(-1, z_past.shape[-1])
        # stack it back into a matrix and return
        return np.vstack([i_past_reshaped, j_past_reshaped, z_past_reshaped])


    def __residual(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:   #private
        """
        Computes the residuals of a variable x by regressing a
        conditioning set z out of it

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


    def __ideal_lp(self, M):
        amp = np.ones(M)
        amp[int(self.f_c * M / self.f_s):-int(self.f_c * M / self.f_s)] = 0
        phase = np.zeros(M)
        H_f = amp * np.exp(1j * phase)
        h_n = np.fft.fftshift(np.real(np.fft.ifft(H_f)))
        return h_n


    def fit_rising(self, X: np.ndarray, idx: np.ndarray, verbose=1):
        """
        Fits c-gc to data

        Args:
            data: array-like of shape (n_neur, T)
                where `n_neur` is the number of neurons or variables
                and `T` is the time or number of samples

        Returns:
            self: object
                Returns the instance itself
        """
        self.n_neur = X.shape[0]
        corr = np.zeros((len(idx) * self.n_pasts, len(idx)))
        pVal_corr = np.zeros((len(idx) * self.n_pasts, len(idx)))
        inv_corr = np.zeros_like(corr)
        pVal_inv_corr = np.zeros_like(corr)
        for i in tqdm(range(len(idx) * self.n_pasts)):
            for j in range(len(idx)):
                if i < X.shape[0]:
                    same_idx = sorted(set(idx[i]).intersection(idx[j]))   # combine()
                else:
                    i_= i%len(idx)
                    same_idx = sorted(set(idx[i_]).intersection(idx[j])) # combine()
                if len(same_idx) > 0.1 * len(same_idx):
                    dat = self.shift_data(X[:, same_idx], self.n_pasts)
                    corr[i, j] = np.abs(np.corrcoef(dat[i], dat[j])[1, 0])
                    pVal_corr[i, j] = self.perm_test_(dat[i], dat[j], self.n_perm)

                    x = dat[i]
                    y = dat[j]
                    z = self.get_conditioning_set(X[:, same_idx], self.n_pasts, i, j)

                    x_ = self.__residual(x, z)
                    y_ = self.__residual(y, z)

                    inv_corr[i, j] = np.abs(np.corrcoef(x_, y_)[1,0])
                    pVal_inv_corr[i, j] = self.perm_test_(x_, y_, self.n_perm)

                else:
                    corr[i,j], pVal_corr[i,j] = 0,1
                    inv_corr[i,j], pVal_inv_corr[i,j] = 0,1

        self.corr_, self.pVal_corr_ = corr, pVal_corr
        self.inv_corr_, self.pVal_inv_corr_ = inv_corr, pVal_inv_corr

        return self









def cross_corr_(x, y, n_lags):
    return np.abs(np.corrcoef(x[:-n_lags], y[n_lags:])[1,0])  



def cutt_mvg(arr,win_len):
    row_ = moving_avg(arr,window_size=5)
    x_ = np.roll(row_,np.random.randint(30,len(arr)-30,1)) # np.random.randint(30,len(arr)-30,1) Used 10 for proper acccessment
    j = np.diff(x_)> np.mean(np.diff(row_)>0) # filtering out small rises that are less than 0  
    idx = np.where(j!=0)[0]
    return np.pad(row_,(0,win_len-1),'constant')[idx]



def cutt_lp(arr,f_c,f_s,M):
    h_n = ideal_lp(f_c,f_s,M)
    conv = np.convolve(arr,h_n,'same')
    j = np.diff(conv) > np.mean(np.diff(conv)>0.5)/10     # /5
    idx = np.where(j>0)[0]
    return arr[idx],idx      # conv[idx] is the squashed vector  # arr ==> conv



def idx_check_(row):
    row_list, new, temp = list(row),[],[]
    for i in range(len(row_list)-1):
        if row_list[i]+1 == row_list[i+1]:
        # grow this list
            temp.append(row_list[i])
        else:
        # add the element that is not the same as the next element, then create a new list
            temp.append(row_list[i])
            new.append(temp.copy())
            temp = []
    else:
      # when the for loop finishes, there is one element left over. This else clause will run when the for loop finishes
        temp.append(row_list[-1])
        new.append(temp)
    return new



def combine(list_of_indices):
    new_list_of_indices = []
    for i in range(len(list_of_indices)-1):
        idx = list_of_indices[i]
        new_list_of_indices.append(idx)
        if idx + 2 == list_of_indices[i+1]:
            new_list_of_indices.append(idx+1)
    new_list_of_indices.append(list_of_indices[-1])
    return new_list_of_indices



def diff_(inf_rising,inf):
    mat = inf_rising==inf
    mat = inf_rising.copy()
    mat[inf_rising==inf] = False
    return mat
#####################################################
#####################################################
