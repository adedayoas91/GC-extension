#!/usr/bin/env python3
# coding: utf-8

import os
from typing import Self
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import mat73
from numba import jit, njit, vectorize
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


#####################################################
#####################################################
### Rising Flanks
#####################################################
#####################################################

def new_GC_analysis(X,idx,n_past,n_perm,num):
    corr,pVal_corr = np.zeros((len(idx)*n_past, len(idx))),np.zeros((len(idx)*n_past,len(idx)))
    inv_corr,pVal_inv_corr = np.zeros_like(corr), np.zeros_like(corr)
    for i in tqdm(range(len(idx)*n_past)):
        for j in range(len(idx)):
            if i < X.shape[0]:
                same_idx = sorted(set(idx[i]).intersection(idx[j]))   # combine()
            else:
                i_= i%len(idx)
                same_idx = sorted(set(idx[i_]).intersection(idx[j])) # combine()
            if len(same_idx)>num:
                dat = prep_data(X[:,same_idx],n_past)
                corr[i,j],pVal_corr[i,j] = np.abs(np.corrcoef(dat[i],dat[j])[1,0]),perm_test_(dat[i],dat[j],n_perm)
                x,y,z = dat[i],dat[j], self.conditioning_set(X[:,same_idx],n_past,i,j)
                x_,y_ = residual(x,z),residual(y,z)
                inv_corr[i,j],pVal_inv_corr[i,j] = np.abs(np.corrcoef(x_,y_)[1,0]),perm_test_(x_,y_,n_perm)  
            else:
                corr[i,j],pVal_corr[i,j] = 0,1
                inv_corr[i,j],pVal_inv_corr[i,j] = 0,1
    return corr,pVal_corr,inv_corr,pVal_inv_corr



def moving_avg(arr,window_size=3):
    i,moving_averages = 0, []
    while i<(len(arr)-window_size+1):
        window = arr[i:i + window_size]
        window_average = round(sum(window) / window_size, 4)
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def ideal_lp(f_c,f_s,M):
    amp = np.ones(M)
    amp[int(f_c*M/f_s):-int(f_c*M/f_s)] = 0
    phase = np.zeros(M)
    H_f = amp*np.exp(1j*phase)
    h_n = np.fft.fftshift(np.real(np.fft.ifft(H_f)))
    return h_n


@jit(nopython=True)
def perm_test_(x,y,N):
    count, corr_1 = 0, np.corrcoef(x,y)[1,0]
    val = np.random.randint(5, len(x)-8, N)     # change 4 back to 30 and len(x)-30
    for j in range(N):
        x_copy = np.copy(x)
        x_ = np.roll(x_copy,val[j])
        corr_2 = np.corrcoef(x_,y)[1,0]
        if np.abs(corr_2) >= np.abs(corr_1):
            count+=1
    return count/N



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
