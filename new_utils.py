#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import mat73
from numba import jit, njit, vectorize
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


###################################################################
## Analysis
###################################################################

@njit
def cross_corr(x, y, n_lags):
    """Returns the absolute values of cross correlation of 2 variables with any specified lag window

    Parameters:
    x (array like): array of a variable of any length
    y (array like): array of a variable of same length as x
    n_lags (int): lag window size (function takes both positive and negative lag of the same window)

    Returns:
    cross correlation of 2 variable (array like), each entry of the array, the correlationat each lag window.

   """

    lags = np.arange(-n_lags, n_lags+1)
    corr_coef = np.zeros(len(lags))
    for i in range(len(corr_coef)):
        if lags[i]< 0:
            corr_coef[i] = np.corrcoef(x[:-np.abs(lags[i])], y[np.abs(lags[i]):])[1,0]
        elif lags[i]==0:
            corr_coef[i] = np.corrcoef(x, y)[1,0]
        else:
            corr_coef[i] = np.corrcoef(x[lags[i]:], y[:-lags[i]])[1,0]

    return np.abs(corr_coef)



@jit(nopython=True)
def perm_test(x, y, shuffle):
    """Performs the naive permutation of a random variable and find its p_value in the null distribution

    Assumptions:
    Time series exhibit both exchangeability and is iid

    Parameters:
    x (array like): array of a variable of any length
    y (array like): array of a variable of same length as x
    shuffle (int): number of permutations desired

    Returns:
    p_values (array like):

    """

    count = 0
    corr_1 = np.corrcoef(x,y)[1,0]
    for j in range(shuffle):
        x_copy = np.copy(x)
        np.random.shuffle(x_copy)
        corr_2 = np.corrcoef(x_copy,y)[1,0]
        if np.abs(corr_2) >= np.abs(corr_1):
            count+=1
    p_value = count/shuffle
    return p_value



@jit(nopython=True)
def perm_test_shift(x, y,shuffle):
    """Performs the naive permutation of a random variable and find its p_value in the null distribution

    Assumptions:
    Time series exhibit exchangeability but not iid

    Parameters:
    x (array like): array of a variable of any length
    y (array like): array of a variable of same length as x
    shuffle (int): number of permutations desired

    Returns:
    p_values (array like):

    """
    count = 0
    corr_1 = np.corrcoef(x,y)[1,0]
    val = np.random.randint(30, len(x)-30,shuffle)
    for j in range(shuffle):
        x_copy = np.copy(x)
        x_ = np.roll(x_copy,val[j])
        corr_2 = np.corrcoef(x_,y)[1,0]
        if np.abs(corr_2) >= np.abs(corr_1):
            count+=1
    p_value = count/shuffle
    return p_value



def correlation(data,n_perm,n_lag=None,iid=bool):
    if n_lag == None:
        corr = np.abs(np.corrcoef(data))
        pVal_corr = np.zeros_like((corr))
        for i in range(0,data.shape[0]):
            for j in range(i,data.shape[0]):
                if iid==True:
                    pVal_corr[i,j] = perm_test(data[i,:],data[j,:],n_perm)
                    pVal_corr[j,i] = pVal_corr[i,j]
                else:
                    pVal_corr[i,j] = perm_test_shift(data[i,:],data[j,:],n_perm)
                    pVal_corr[j,i] = pVal_corr[i,j]

        return corr, pVal_corr, pVal_corr

    else:
        n_var = data.shape[0]
        lags = np.arange(-n_lag,n_lag+1).astype(int)
        corr = np.empty([n_var,n_var])
        pVal_corr = np.zeros_like(corr)
        max_corr_lag = np.zeros((n_var,n_var))
        for i in range(0,data.shape[0]):
            for j in range(i,data.shape[0]):
                cross_corr_results = cross_corr(data[i,:],data[j,:], n_lag)
                corr[i,j] = np.max(cross_corr_results)
                corr[j,i] = corr[i,j]
                max_corr_lag[i,j] = lags[np.argmax(cross_corr_results)].astype(int)
                max_corr_lag[j,i]= max_corr_lag[i,j]
                if iid==True:
                    if max_corr_lag[i,j]< 0:
                        pVal_corr[i,j] = perm_test(data[i,:-np.abs(max_corr_lag[i,j])], data[j,np.abs(max_corr_lag[i,j]):],n_perm)
                        pVal_corr[j,i] = pVal_corr[i,j]
                    elif max_corr_lag[i,j] == 0:
                        pVal_corr[i,j] = perm_test(data[i,:], data[j,:],n_perm)
                        pVal_corr[j,i] = pVal_corr[i,j]
                    else:
                        pVal_corr[i,j] = perm_test(data[i,max_corr_lag[i,j]:], data[j,:-max_corr_lag[i,j]],n_perm)
                        pVal_corr[j,i] = pVal_corr[i,j]

                else:
                    if max_corr_lag[i,j] < 0:
                        pVal_corr[i,j] = perm_test_shift(data[i,:-np.abs(int(max_corr_lag[i,j]))], data[j,np.abs(int(max_corr_lag[i,j])):],n_perm)
                        pVal_corr[j,i] = pVal_corr[i,j]
                    elif max_corr_lag[i,j] == 0:
                        pVal_corr[i,j] = perm_test_shift(data[i,:], data[j,:],n_perm)
                        pVal_corr[j,i] = pVal_corr[i,j]
                    else:
                        pVal_corr[i,j] = perm_test_shift(data[i,int(max_corr_lag[i,j]):], data[j,:-int(max_corr_lag[i,j])],n_perm)
                        pVal_corr[j,i] = pVal_corr[i,j]

        return corr, pVal_corr, max_corr_lag



def inv_correlation(data,n_perm,n_lag=None,iid=bool):
    """
    Returns both naive and shifted permuted significant inverse correlation matrix at zero lag.      # REMEMBER TO ADD MULTI LINE COMMENTS

    Parameters:
    data (array like, matrix): empirical data
    n_perm (int): desired number of permutations.
    alpha (float): statistical significance level (usually 0.05)

    Returns:
    isig_naive (array like, matrix): significant inverse correlation matrix, based on the naive permmutation
    isig_shifted (array like, matrix): significant inverse correlation matrix, based on the shifted permmutation

    """
    corr = np.abs(np.corrcoef(data))
    n_var = corr.shape[0] # number of neurons
    inv_corr = np.empty([n_var,n_var])
    pVal_inv_corr = np.zeros((n_var,n_var))
    if n_lag == None:
        for i in range(0,n_var):
            for j in range(i,n_var):
                x = data[i,:]
                y = data[j,:]
                z = np.delete(data,[i,j],axis=0)
                x_residual = residual(x,z)
                y_residual = residual(y,z)
                if iid==True:
                    pVal_inv_corr[i,j] = perm_test(x_residual,y_residual,n_perm)
                    pVal_inv_corr[j,i] = pVal_inv_corr[i,j]
                else:
                    pVal_inv_corr[i,j] = perm_test_shift(x_residual,y_residual,n_perm)
                    pVal_inv_corr[j,i] = pVal_inv_corr[i,j]
                inv_corr[i,j] = np.abs(np.corrcoef(x_residual,y_residual)[1,0])
                inv_corr[j,i]= inv_corr[i,j]
        return inv_corr, pVal_inv_corr

    else:
        corr_coef_no_lag, max_ccoef_mat, l = correlation(data,n_lag)
        l = l.astype(int)
        for a in range(0,n_var):
            for b in range(a,n_var):
                # for l in lags  ----> such that i list all lag values and i can run the loop on each lag values:
                if l[a,b]< 0:
                    x = data[a,:-np.abs(l[a,b])]
                    y = data[b,np.abs(l[a,b]):]
                    z = np.delete(data[:,np.abs(lags[i]):],[a,b],0)

                elif l[a,b]==0:
                    x = data[a,:]
                    y = data[b,:]
                    z = np.delete(data,[a,b],0)
                else:
                    x = data[a,l[a,b]:]
                    y = data[b,:-l[a,b]]
                    z = np.delete(data[:,:-l[a,b]],[a,b],0)
                x_residual = residual(x,z)
                y_residual = residual(y,z)
                inv_corr[a,b] = np.max(cross_corr(x_residual,y_residual,n_lag))
                inv_corr[b,a] = inv_corr[a,b]
                if iid==True:
                    pVal_inv_corr[a,b] = perm_test(x_residual,y_residual,n_perm)
                    pVal_inv_corr[b,a] = pVal_inv_corr[a,b]
                else:
                    pVal_inv_corr[a,b] = perm_test_shift(x_residual,y_residual,n_perm)
                    pVal_inv_corr[b,a] = pVal_inv_corr[a,b]

        return inv_corr, pVal_inv_corr



def inv_correlation_new(data,n_perm,n_lag=None,iid=bool):
    """Returns both naive and shifted permuted significant inverse correlation matrix at zero lag.

    Parameters:
    data (array like, matrix): empirical data
    n_perm (int): desired number of permutations.
    alpha (float): statistical significance level (usually 0.05)

    Returns:
    isig_naive (array like, matrix): significant inverse correlation matrix, based on the naive permmutation
    isig_shifted (array like, matrix): significant inverse correlation matrix, based on the shifted permmutation
    """
    corr = np.abs(np.corrcoef(data))
    n_var = corr.shape[0] # number of neurons
    inv_corr = np.empty([n_var,n_var])
    pVal_inv_corr = np.zeros((n_var,n_var))
    if n_lag == None:
        for i in range(0,n_var):
            for j in range(i,n_var):
                x = data[i,:]
                y = data[j,:]
                z = np.delete(data,[i,j],axis=0)
                x_residual = residual(x,z)
                y_residual = residual(y,z)
                if iid==True:
                    pVal_inv_corr[i,j] = perm_test(x_residual,y_residual,n_perm)
                    pVal_inv_corr[j,i] = pVal_inv_corr[i,j]
                else:
                    pVal_inv_corr[i,j] = perm_test_shift(x_residual,y_residual,n_perm)
                    pVal_inv_corr[j,i] = pVal_inv_corr[i,j]
        return pVal_inv_corr

    else:
        lags = np.arange(-n_lag,n_lag+1,dtype='int')
        #inv_corr = np.empty([n_var,n_var,len(lags)])
        pVal_inv_corr = np.zeros((n_var,n_var,len(lags)))
        corr_coef_no_lag, max_ccoef_mat, _ = correlation(data,n_lag)
        for a in range(0,n_var):
            for b in range(a,n_var):
                for i, l in enumerate(lags):      # ----> such that lags is a list of all lag values in the window of lag chosen
                    if l < 0:
                        x = data[a,:-np.abs(l)]
                        y = data[b,np.abs(l):]
                        z = np.delete(data[:,np.abs(l):],[a,b],0)

                    elif l==0:
                        x = data[a,:]
                        y = data[b,:]
                        z = np.delete(data,[a,b],0)
                    else:
                        x = data[a,l:]
                        y = data[b,:-l]
                        z = np.delete(data[:,:-l],[a,b],0)
                    x_residual = residual(x,z)
                    y_residual = residual(y,z)
                    if iid==True:
                        pVal_inv_corr[a,b,i] = perm_test(x_residual,y_residual,n_perm)
                        pVal_inv_corr[b,a,i] = pVal_inv_corr[a,b,i]
                    else:
                        pVal_inv_corr[a,b,i] = perm_test_shift(x_residual,y_residual,n_perm)
                        pVal_inv_corr[b,a,i] = pVal_inv_corr[a,b,i]

        return pVal_inv_corr



def residual(x,z):
    """Returns the residual of a given neuron after regressing out the effects of other neurons
    uses the equation X = x - np.dot(coefs,z)-intercept

    Parameters:
    data (array like, matrix): empirical data
    N (int): Number of traces to be simulated for individual neuron

    Returns:
    A 3-D matrix of all the simulated traces with shape [no_of_neurons X no_of_timepoints X N]
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(z.T,x)
    coefs = model.coef_
    intercept = model.intercept_
    residual = x - np.dot(coefs,z)-intercept
    return residual



def analyse(data,n_perm,alpha,n_lag=None,iid=bool,return_pValue=bool):
    corr,pVal_corr, _ = correlation(data,n_perm,n_lag,iid=bool)
    inv_corr,pVal_inv_corr = inv_correlation_new(data,n_perm,n_lag,iid=bool)
    sig_corr = np.multiply(corr,pVal_corr<=alpha)
    sig_inv_corr = np.multiply(inv_corr,pVal_inv_corr<=alpha)
    inferred_adj_mtx = np.logical_and(sig_corr,sig_inv_corr)
    np.fill_diagonal(inferred_adj_mtx,0)
    if return_pValue == True:
        return corr,pVal_corr,pVal_inv_corr
    else:
        return inferred_adj_mtx



def new_analyse(data,n_perm,n_lag,iid=bool):
    corr,pVal_corr, _ = correlation(data,n_perm,n_lag,iid=bool)
    inv_corr, pVal_inv_corr = inv_correlation(data,n_perm,n_lag,iid=bool)   # change to inv_correlation_new
    return corr,pVal_corr,inv_corr,pVal_inv_corr


##################################################################
## Evaluation
###################################################################

def confusion_matrix(inferred, A):
    """
    To compute metrics such as true positive, true negatives, false positives, false negatives

    Parameters
    inferred (matrix) - Inferred adjacency matrix returned by inf_adj_mtx()
    A (matrix) - Initial adjacency matrix

    Return
    A confusion matrix of the form [[TP,FN],
                                    [FP,TN]]
    """
    TP_inf = np.sum(np.logical_and(A != 0,inferred!=0))
    FN_inf = np.sum(np.logical_and(A != 0,inferred==0))
    FP_inf = np.sum(np.logical_and(A == 0,inferred!=0))
    TN_inf = np.sum(np.logical_and(A == 0,inferred==0))
    return np.array([[TP_inf,FN_inf],
                     [FP_inf,TN_inf]])



def apr_metrics(confusion_matrix):
    confusion_matrix = confusion_matrix.flatten()
    accuracy = (confusion_matrix[0] + confusion_matrix[3])/(np.sum(confusion_matrix))
    precision = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[2])
    recall = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[1])
    FPR = confusion_matrix[2]/(confusion_matrix[2]+confusion_matrix[3])
    return np.array([accuracy, precision, recall, FPR])



def perm_func(n_perm, inferred, adj_mtx):
    evals = []
    for i in range(n_perm):
        perm_ind = np.arange(inferred.shape[0])
        np.random.shuffle(perm_ind)
        adj_mtx_permuted = adj_mtx[perm_ind,:][:,perm_ind]
        evals.append(confusion_matrix(inferred, adj_mtx_permuted).flatten())
    return np.asarray(evals) ## for the plots, its better to flatten from here to make it easier



def compute_p_values(perm_results, conf_mtx):
    conf_mtx_flatten = conf_mtx.flatten()
    p_values = []
    p_values.append(np.sum(perm_results[:,0]>conf_mtx_flatten[0])/perm_results.shape[0])
    p_values.append(1-np.sum(perm_results[:,1]>conf_mtx_flatten[1])/perm_results.shape[0])
    p_values.append(1-np.sum(perm_results[:,2]>conf_mtx_flatten[2])/perm_results.shape[0])
    p_values.append(np.sum(perm_results[:,3]>conf_mtx_flatten[3])/perm_results.shape[0])
    return p_values



def plot_dists(conf_mtxs,inferred,adj_mtx):
    fig,ax = plt.subplots(1,4, figsize=(16,4))
    # evals = conf_mtxs[i].flatten()
    for i in range(conf_mtxs.shape[1]):
        ax[i].hist(conf_mtxs[:,i])
        if i == 0:
            ax[i].vlines(x=np.sum(np.logical_and(adj_mtx != 0,inferred!=0)),ymin=0,ymax=conf_mtxs.shape[0]//2, color='r')
            ax[i].set_title('True Positives')
        elif i == 1:
            ax[i].vlines(x=np.sum(np.logical_and(adj_mtx != 0,inferred==0)),ymin=0,ymax=conf_mtxs.shape[0]//2, color='r')
            ax[i].set_title('False Negatives')
        elif i == 2:
            ax[i].vlines(x=np.sum(np.logical_and(adj_mtx==0, inferred!=0)),ymin=0,ymax=conf_mtxs.shape[0]//2, color='r')
            ax[i].set_title('False Positives')
        else:
            ax[i].vlines(x=np.sum(np.logical_and(adj_mtx==0, inferred==0)),ymin=0,ymax=conf_mtxs.shape[0]//2, color='r')
            ax[i].set_title('True Negatives')
    plt.tight_layout()


###################################################################
## Simulation
###################################################################

def simulate_data(A,m,iid=bool):
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
        for i, row in enumerate(X[:-1]):
            X[i+1] = A @ X[i] + 2*np.random.normal(0,0.25,A.shape[0])

    return X.T


###################################################################
## AR models
###################################################################

def modeller(data, n):
    """Autoregressive (AR) model of a time series.

    Parameters:
    data (array like): array of a variable of any length
    n (int): desired order of AR model

    Returns:
    model (object): trained AR model
    rmse: root mean square error of the model

    """

    # split data into train and  test
    length = int(0.8*len(data))
    train, test = data[:length], data[length:]
    X = np.zeros(shape=(train.shape[0]-n, n))
    X_test = np.zeros(shape=(test.shape[0]-n,n))
    labels = train[n:].copy()
    test_labels = test[n:].copy()

    # forming matrix to train model
    for t in range(X.shape[0]):
        X[t, :] = train[t:t+n]

    for j in range(X_test.shape[0]):
        X_test[j,:] = test[j:j+n]
    # adding a column of ones
    X_full = np.hstack([X, np.ones(shape=X.shape[0]).reshape(-1,1)])
    x_test_full = np.hstack([X_test, np.ones(shape=X_test.shape[0]).reshape(-1,1)])
    # training models
    model = LinearRegression(fit_intercept=True)
    model.fit(X_full, labels)

    prediction = model.predict(x_test_full)

    rmse = np.sqrt(mean_squared_error(test_labels, prediction))

    return model, rmse



def simulate(data,model,rmse,n):
    """Traces simulato, simulates traces using trained AR model

    Parameters:
    data (array like): array of a variable of any length
    model (object): trained AR model
    rmse: root mean square error of the model
    n (int): desired order of AR model

    Returns:
    Simulated time series of same length as original time series

    """
    a = len(data)
    z_ = data
    n+=1
    for t in range(len(data)-n,len(data)*2-n):
        new = model.predict(z_[t:].reshape(1,-1)) + np.random.normal(0,rmse,1)
        if new<0:
            new = 0
        z_ = np.hstack((z_,new))
        sim_traces = z_[a:]
    return sim_traces



def simulate_traces(data,N, n):
    """Uses function modeller() and simulate(), see documentations; to simulate N traces of each neurons in the data.

    Parameters:
    data (array like, matrix): empirical data
    N (int): Number of traces to be simulated for individual neuron

    Returns:
    A 3-D matrix of all the simulated traces with shape [no_of_neurons X no_of_timepoints X N]
    """

    all_traces = np.zeros((data.shape[0],data.shape[1],N))

    for i in range(data.shape[0]):
        for a in range(N):
            model, rmse =  modeller(data[i,:],n)
            all_traces[i,:,a] = simulate(data[i,:],model,rmse,n)

    return all_traces



def plot_simtraces(data,simulated,m,n):
    """Plot randomly selected simulated traces from the result of simultated_traces() to compare the simulated with the original traces

    Parameters:
    data (array like, matrix): empirical data
    simulated (matrix): N simulated traces of all/single neuron (3D matrix in case of whole population of neurons)
    m (int): Number of traces of neuron 'n' to be simulated to be visualized
    n (int): the index of neuron whose traces are to be plotted

    Returns:
    A plot of original and 'm' simulated traces.
    """

    fig,ax = plt.subplots(m+1,1,figsize=(14,m*3))
    ax[0].plot(data[n,:])
    ax[0].axhline(data[n,:].mean(), color='r', alpha=0.5, linestyle='--')
    ax[0].set_title('original traces of neuron {}'.format(n))
    for i in range(1,len(ax)):
        rand = np.random.randint(0,simulated.shape[2],1,dtype=int)[0]
        ax[i].plot(simulated[n,:,rand])
        ax[i].set_title('simulted traces {} of neuron {}'.format(rand, n))
        ax[i].axhline(simulated[n,:,rand].mean(),color='red', linestyle='--')
    plt.tight_layout()



def AR_perm(data,all_traces,alpha,n_lags=None):
    """Computes naive permutation using correlation of any 2 variable pairs as test statistic.
    The function returns p_values of the permutation test both at zero and desired lag window values (lag_idx*).
    *see documentation of max_corr_lag from correlation()

    Parameters:
    data (array like, matrix): array of a variable of any length. (Variables are aligned on the rows)
    all_traces (3-D matrix of all simulated traces with shape [no_of_neurons,trace_len,n_perm]):
    n_lags (int): desired number of permutations.

    Returns:
    pval_no_lag (array like, matrix): matrix of of p_values at zero lags
    pval_lag (array like, matrix): matrix of of p_values at maximum correlation lag index.

    """
    n_var = data.shape[0]
    n_perm = all_traces.shape[2]
    pValues_AR = np.empty([n_var,n_var])


    if n_lags == None:
        corr,pVal, _ = correlation(data,n_perm,n_lags,iid=False)
        for n in range(0,n_var):
            for m in range(n,n_var):
                count = 0
                for k in range(n_perm):
                    ccor_2 = np.abs(np.corrcoef(all_traces[n,:,k],data[m,:])[1,0])
                    if ccor_2 >= corr[n,m]:
                        count+=1
                pValues_AR[n,m] = count/n_perm
                pValues_AR[m,n]=pValues_AR[n,m]
        return corr, pValues_AR


    else:
        corr, pVal_corr, max_corr_lag = correlation(data,n_perm,n_lags,iid=False)
        for n in range(0,n_var):
            for m in range(n,n_var):
                count = 0
                for k in range(n_perm):
                    ccor_2 = np.max(cross_corr(all_traces[n,:,k], data[m,:],n_lags))
                    if ccor_2 >= corr[n,m]:
                        count+=1
                pValues_AR[n,m] = count/n_perm
                pValues_AR[m,n]=pValues_AR[n,m]
        return corr, pValues_AR


#######################################################
## pre processing C elegans data
#######################################################

class Database:

    def __init__(self, data_set_no=2):
        data_dict = mat73.loadmat('NoStim_Data.mat')
        data  = data_dict['NoStim_Data']

        deltaFOverF_bc = data['deltaFOverF_bc'][data_set_no]
        derivatives = data['derivs'][data_set_no]
        NeuronNames = data['NeuronNames'][data_set_no]
        fps = data['fps'][data_set_no]
        States = data['States'][data_set_no]


        self.states = np.sum([n*States[s] for n, s in enumerate(States)], axis = 0).astype(int) # making a single states array in which each number corresponds to a behaviour
        self.state_names = [*States.keys()]
        self.neuron_traces = np.array(deltaFOverF_bc).T
        self.derivative_traces = derivatives['traces'].T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

        f = open('readme.txt', 'r')
        self.DESCR = f.read()
        f.close()
        '''
        #Sort the data according to the clustering dendogram (only for dataset 3, as of now)
        self.neuron_traces = self.neuron_traces[sort_indices]
        self.derivative_traces = self.derivative_traces[sort_indices]
        self.NeuronNames = self.NeuronNames[sort_indices]
        '''
        ## Creating dictionary of identified neurons and their indices
        #self.neuron_id = {}
        #for n, i in enumerate(self.NeuronNames):
        #    if type(i) == list:
        #        self.neuron_id[i[0]]=n



def plot_raster(neuron_traces, derivative_traces):
    fig, ax = plt.subplots(2,1, figsize=(15,10))
    plt0 = ax[0].imshow(neuron_traces, aspect="auto", vmin=0, vmax=1)
    #ax[0].set_yticks(np.arange(neuron_names.shape[0]))
    #ax[0].set_yticklabels(neuron_names)
    fig.colorbar(plt0, ax=ax[0])
    plt1 = ax[1].imshow(derivative_traces, cmap='seismic', aspect="auto", vmin=-0.25, vmax=0.25)
    #ax[1].set_yticks(np.arange(neuron_names.shape[0]))
    #ax[1].set_yticklabels(neuron_names)
    fig.colorbar(plt1, ax=ax[1])
    plt.show()



def dendogram(classifier):
    ## Dendogram
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(classifier.children_.shape[0])
    n_samples = len(classifier.labels_)
    for i, merge in enumerate(classifier.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([classifier.children_, classifier.distances_, counts]).astype(float)
    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, truncate_mode='level')
    ## Sorting: The features are ordered according to the order of the leaves in the dendogram
    sort_indices = R['leaves']
    return sort_indices



def plot_Inferred_3D(inferred,centers,arr):
    x_val,y_val,z_val = xyz_maker(inferred,centers)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111,projection="3d")
    ax.scatter(centers[:,0],centers[:,1],centers[:,2],color='red',s=10,marker='.')
    ax.scatter(centers[arr,0],centers[arr,1],centers[arr,2],color='green',s=15,marker='*')
    for a in range(len(x_val)):
        ax.plot(x_val[a],y_val[a],z_val[a],lw=0.7,alpha=.7)
    ax.grid(False)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('z-axis')
    return fig



def xyz_maker(inferred,topography):
    """
    A func to make x and y to be used to plot prec in the form understandable to mapping()...
    # topography is centers
    """
    t = np.transpose(np.where(inferred>0))
    point_1 = np.zeros_like(t)
    point_2 = np.zeros_like(t)

    for i in range(len(t)):
        point_1[i]=topography[t[i,0],[0,1]]
        point_2[i]=topography[t[i,1],[0,1]]

    x_val,y_val,z_val = [],[],[]
    for i in range(len(point_1)):
        x_val.append([point_1[i,0],point_2[i,0]])
        y_val.append([point_1[i,1],point_2[i,1]])
        z_val.append([topography[t[i,0],2],topography[t[i,1],2]])

    return x_val,y_val,z_val
