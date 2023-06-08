#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from numba import jit, njit, vectorize
import nrrd
from scipy.stats import zscore



# cross correlation
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

    lags = np.arange(-n_lags+1, n_lags)
    corr_coef = np.zeros(len(lags))
    for i in range(len(corr_coef)):
        if lags[i]< 0:
            corr_coef[i] = np.corrcoef(x[:-np.abs(lags[i])], y[np.abs(lags[i]):])[1,0]
        elif lags[i]==0:
            corr_coef[i] = np.corrcoef(x, y)[1,0]
        else:
            corr_coef[i] = np.corrcoef(x[lags[i]:], y[:-lags[i]])[1,0]

    return np.abs(corr_coef)


def correlation(data,n_lags):
    """Computes correlation of a given data both at zero and a desired lag window. Takes both positive and negative lags.

    Parameters:
    data (array like, matrix): array of a variable of any length. (Variables are aligned on the rows)
    n_lags (int): desired lag window size to be taken.

    Returns:
    corr_coef_no_lag (array like, matrix): Correlation matrix of data at zero lags
    max_ccoef_mat (array like, matrix): Correlation matrix of data at a specified maximum lag window (n_lags)
    max_corr_lag (array like, matrix): Matrix of lag at which maximum correlation occured.

    """

    # cross correlation at zero lag using the np.corrcoef()
    corr_coef_no_lag = np.abs(np.corrcoef(data))
    #corr_coef_no_lag = np.triu(corr_coef_no_lag,k=0) # selecting the upper triangle

    n_var = corr_coef_no_lag.shape[0]
    ##########################################################
    # computing cross correlation of all neuron with lag of 20
    # selected the max correlations at which ever lag it occurs
    ##########################################################

    max_ccoef_mat = np.zeros((n_var,n_var))
    max_corr_lag = np.zeros((n_var,n_var))
    lags = np.arange(-n_lags+1,n_lags).astype(int)
    for n in range(0,n_var):
        for i in range(n,n_var):
            ccor = cross_corr(data[n, :], data[i, :], n_lags)
            max_ccoef_mat[n,i] = np.max(ccor)
            max_ccoef_mat[i,n] = max_ccoef_mat[n,i]
            max_corr_lag[n,i] = lags[np.argmax(ccor)]
            max_corr_lag[i,n]= max_corr_lag[n,i]

    max_corr_lag = max_corr_lag.astype(int)

    return corr_coef_no_lag, max_ccoef_mat, max_corr_lag


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
def perm_test_shift(x, y, shuffle):
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
        if corr_2 >= corr_1:
            count+=1
    p_value = count/shuffle
    return p_value


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
    # adding a column of 1s to the matrix for bias
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


def naive_permutation(data,n_perm,lag_idx):
    """Computes naive permutation using correlation of any 2 variable pairs as test statistic.
    The function returns p_values of the permutation test both at zero and desired lag window values (lag_idx*).
    *see documentation of max_corr_lag from correlation()

    Parameters:
    data (array like, matrix): array of a variable of any length. (Variables are aligned on the rows)
    n_perm (int): desired number of permutations.

    Returns:
    pval_no_lag (array like, matrix): matrix of of p_values at zero lags
    pval_lag (array like, matrix): matrix of of p_values at maximum correlation lag index.

    """

    n_var = data.shape[0]
    pval_no_lag = np.empty([n_var,n_var])
    for i in range(0,n_var):
        for j in range(i,n_var):
            pval_no_lag[i,j] = perm_test(data[i,:], data[j,:],n_perm)
            pval_no_lag[j,i] = pval_no_lag[i,j]

    # Computing p_values at window lag of maximum correlations
    pval_lag = np.empty([n_var,n_var])
    for a in range(0,n_var):
        for b in range(a,n_var):
            if lag_idx[a,b]< 0:
                pval_lag[a,b] = perm_test(data[a,:-np.abs(lag_idx[a,b])], data[b,np.abs(lag_idx[a,b]):],n_perm)
            elif lag_idx[a,b] == 0:
                pval_lag[a,b] = perm_test(data[a,:], data[b,:],n_perm)
            else:
                pval_lag[a,b] = perm_test(data[a,lag_idx[a,b]:], data[b,:-lag_idx[a,b]],n_perm)
            pval_lag[b,a] = pval_lag[a,b]

    return pval_no_lag, pval_lag


def shifted_permutation(data,n_perm,lag_idx):
    """Computes shifted time series permutation using correlation of any 2 variable pairs as test statistic.
    The function returns p_values of the permutation test both at zero and desired lag window values (lag_idx*).
    *see documentation of max_corr_lag from correlation()

    Parameters:
    data (array like, matrix): array of a variable of any length. (Variables are aligned on the rows)
    n_perm (int): desired number of permutations.

    Returns:
    pval_no_lag (array like, matrix): matrix of of p_values at zero lags
    pval_lag (array like, matrix): matrix of of p_values at maximum correlation lag index.

    """

    n_var = data.shape[0]
    pval_no_lag = np.empty([n_var,n_var])
    for i in range(0,n_var):
        for j in range(i,n_var):
            pval_no_lag[i,j] = perm_test_shift(data[i,:], data[j,:],n_perm)
            pval_no_lag[j,i] = pval_no_lag[i,j]

    # Computing p_values at window lag of maximum correlations

    pval_lag = np.empty([n_var,n_var])
    for a in range(0,n_var):
        for b in range(a,n_var):
            if lag_idx[a,b]< 0:
                pval_lag[a,b] = perm_test_shift(data[a,:-np.abs(lag_idx[a,b])], data[b,np.abs(lag_idx[a,b]):],n_perm)
            elif lag_idx[a,b] == 0:
                pval_lag[a,b] = perm_test_shift(data[a,:], data[b,:],n_perm)
            else:
                pval_lag[a,b] = perm_test_shift(data[a,lag_idx[a,b]:], data[b,:-lag_idx[a,b]],n_perm)
            pval_lag[b,a] = pval_lag[a,b]

    return pval_no_lag, pval_lag



def significant(matrix,p_values,alpha):
    """Sets insignificant values in a correlation matrix to zero with respect to a computed p_value matrix. Decision based on selected alpha value.

    Parameters:
    matrix (array like): Correlation matrix
    p_values (array like): Matrix of computed p_values
    alpha = statistical significance level (usually 0.05)

    Returns:
    sig (array like, matrix): An array of significant correlation coefiients
    """
    sig = np.copy(matrix)
    n = p_values.shape[0]
    for i in range(n):
        for j in range(n):
            if p_values[i,j]>alpha:
                sig[i,j] = 0
    np.fill_diagonal(sig,0)
    np.triu(sig,k=0)
    return sig



def mapping(corr_coef,topography,info):
    """Plots the significant links between neurons and map it onto the topography of the fish for a given plane.

    Parameters:
    corr_coef (array like): significant correlation coeficient matrix
    topography (array like): Computed ROI (map of neurons in the plane of interest)

    Returns:
    A plot of the significant links between the cells. THe width of the lines is the correlation coeficient of the 2 cells
    """

    t = np.transpose(np.where(corr_coef>0))
    point_1 = np.zeros_like(t)
    point_2 = np.zeros_like(t)
    for i in range(len(t)):
        point_1[i]=topography[t[i,0]]
        point_2[i]=topography[t[i,1]]
    plt.figure(figsize=(12,6))
    plt.imshow(info.loc[(1,'02')].background,cmap=plt.cm.gist_yarg,vmax=500,origin='lower')
    plt.scatter(topography[:,0],topography[:,1],s=30,color='r')
    plt.title('{} significant links on topography of fish'.format(len(t)))
    for i in range(len(point_1)):
        x_value = [point_1[i,0],point_2[i,0]]
        y_value = [point_1[i,1],point_2[i,1]]
        plt.plot(x_value,y_value,linewidth=corr_coef[t[i,0],t[i,1]])
    plt.colorbar()


############################################################################
### Following functtions are not adapted for general purpose of the toolbox
### they are mainly for easy access of the current state of data
############################################################################

def centre_extractor(df_ana,new_data,a):
    d,e = new_data.loc[a]['fish'],new_data.loc[a]['plane']
    M = len(df_ana.loc[d,e]['x_registered'])
    centers = []
    for i in range(M):
        centers.append([df_ana.loc[d,e]['x_registered'][i],df_ana.loc[d,e]['y_registered'][i]])
    return np.array(centers)


def xy_maker(prec,topography):
    """
    A func to make x and y to be used to plot prec in the form understandable to mapping()...
    # topography is centers
    """
    t = np.transpose(np.where(prec>0))
    point_1 = np.zeros_like(t)
    point_2 = np.zeros_like(t)
    for i in range(len(t)):
        point_1[i]=topography[t[i,0]]
        point_2[i]=topography[t[i,1]]
    x_val,y_val = [],[]
    for i in range(len(point_1)):
        x_val.append([point_1[i,0],point_2[i,0]])
        y_val.append([point_1[i,1],point_2[i,1]])
    return x_val,y_val


def new_mapping(prec,new_data,df_ana,a,background):
    """Plots the significant links between neurons and map it onto the topography of the fish for a given plane.

    Parameters:
    prec (array like): significant correlation coeficient matrix
    df (dataframe): All planes sharing the same reference plane (z_ref_manual)
    df_ana (dataframe): to extract the cell centers using x_y_registered
    a (int): indexing the dataframe at specific z_ref_manual

    Returns:
    A plot of the significant links between the cells. THe width of the lines is the correlation coeficient of the 2 cells
    """
    topography = centre_extractor(df_ana,new_data,a)
    x_val,y_val = xy_maker(prec,topography)
    plt.figure(figsize=(10,6))
    plt.imshow(zscore(background[new_data['z_per_plane'][0]]),origin='lower')
    plt.scatter(topography[:,0],topography[:,1],s=30,color='r')
    plt.plot(x_val,y_val)
    # plt.title('fish{}_plane{}'.format([new_data.loc[a]['fish'],[new_data.loc[a]['plane']]]))

def distance(centers,prec):
    """statistical check for coherence in the distance exhibited by the connections infered

    Parameters:
    centers (array like): list of lists, contain the x and y coordinates of the ROIs
    prec (array like/matrix): significant correlations of the coeficient matrix

    Returns:
    computed euclidean distance between ROIs and the correlation coefficient of each distance.
    """
    loc = np.transpose(np.where(prec>0))
    cor_coef = np.zeros(len(loc))
    dist = np.zeros(len(loc))
    for i in range(len(loc)):
        cor_coef[i] = prec[loc[i,0],loc[i,1]]
        p1 = centers[loc[i,0]]
        p2 = centers[loc[i,1]]
        dist[i] = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    return dist,cor_coef
################################################################################
## Ends here for all the new functions created
################################################################################


def simulate_traces(data,N, n):
    """Uses function modeller() and simulate(), see documentations; to  simulate N traces of each neurons in the data.

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


def AR_perm(data,all_traces,n_lags):
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
    p_values_AR = np.zeros((n_var,n_var))

    # compute the correlation matrix here
    corr_coef_no_lag, max_ccoef_mat, max_corr_lag = correlation(data,n_lags)

    for n in range(n_var):
        for m in range(n_var):
            count = 0
            for k in range(n_perm):
                ccor_2 = np.abs(np.corrcoef(all_traces[n,:,k], data[m,:])[1,0])
                if ccor_2 >= corr_coef_no_lag[n,m]:
                    count+=1
            p_values_AR[n,m] = count/n_perm


    p_values_AR_lag = np.zeros((n_var,n_var))
    for n in range(n_var):
        for m in range(n_var):
            count = 0
            for k in range(n_perm):
                ccor_2 = np.max(cross_corr(all_traces[n,:,k], data[m,:],n_lags))
                if ccor_2 >= max_ccoef_mat[n,m]:
                    count+=1
            p_values_AR_lag[n,m] = count/n_perm

    return p_values_AR, p_values_AR_lag


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


# finiding the inverse correlatioms

def inv_correlation(data,n_perm,alpha):
    """Returns both naive and shifted permuted significant inverse correlation matrix at zero lag.

    Parameters:
    data (array like, matrix): empirical data
    n_perm (int): desired number of permutations.
    alpha (float): statistical significance level (usually 0.05)

    Returns:
    isig_naive (array like, matrix): significant inverse correlation matrix, based on the naive permmutation
    isig_shifted (array like, matrix): significant inverse correlation matrix, based on the shifted permmutation
    """
    # inverse correlation at zero lag
    n_var = data.shape[0]
    inv_ccor1 = np.zeros((n_var,n_var))
    naive_pval = np.zeros((n_var,n_var))
    shifted_pval = np.zeros((n_var,n_var))
    for i in range(0,n_var):
        for j in range(i,n_var):
            x = data[i,:]
            y = data[j,:]
            z = np.delete(data,[i,j],axis=0)
            X = residual(x,z)
            Y = residual(y,z)
            naive_pval[i,j] = perm_test(X,Y,n_perm)
            shifted_pval[i,j] = perm_test_shift(X,Y,n_perm)
            inv_ccor1[i,j] = np.abs(np.corrcoef(X,Y)[1,0])
    isig_naive = significant(inv_ccor1,naive_pval,alpha)
    isig_shifted = significant(inv_ccor1,shifted_pval,alpha)
    return isig_naive, isig_shifted

    # inverse correlation with window lag of 20
def inv_correlation_lag(data,n_perm,n_lags,alpha):
    """Returns both naive and shifted permuted significant inverse correlation matrix a specified lag window


    Parameters:
    data (array like, matrix): empirical data
    n_perm (int): desired number of permutations
    n_lags (int): desired lag window size to be taken.
    alpha (float): statistical significance level (usually 0.05)

    Returns:
    isig_naive_lag (array like, matrix): significant inverse correlation matrix at desired lag window, based on the naive permmutation
    isig_shifted_lag (array like, matrix): significant inverse correlation matrix at desired lag window, based on the shifted permmutation
    """
    n_var = data.shape[0]
    corr_coef_no_lag, max_ccoef_mat, max_corr_lag = correlation(data,n_lags)
    l = max_corr_lag
    inv_ccor2 = np.zeros((n_var,n_var))
    naive_pval_lag = np.zeros((n_var,n_var))
    shifted_pval_lag = np.empty([n_var,n_var])
    for a in range(0,n_var):
        for b in range(a,n_var):
            if l[a,b]< 0:
                x = data[a,:-np.abs(l[a,b])]
                y = data[b,np.abs(l[a,b]):]
                z = np.delete(data[:,:-np.abs(l[a,b])],[a,b],0)

            elif l[a,b]==0:
                x = data[a,:]
                y = data[b,:]
                z = np.delete(data,[a,b],0)
            else:
                x = data[a,l[a,b]:]
                y = data[b,:-l[a,b]]
                z = np.delete(data[:,l[a,b]:],[a,b],0)
            X = residual(x,z)
            Y = residual(y,z)
            inv_ccor2[a,b] = np.max(cross_corr(X,Y,n_lags))
            naive_pval_lag[a,b] = perm_test(X,Y,n_perm)
            shifted_pval_lag[a,b] = perm_test_shift(X,Y,n_perm)
    isig_naive_lag = significant(inv_ccor2,naive_pval_lag,alpha)
    isig_shifted_lag = significant(inv_ccor2,shifted_pval_lag,alpha)
    return isig_naive_lag, isig_shifted_lag


def precision(a,b):
    """Superpose significant correlation and inverse correlation matrices

    Parameters:
    a (array like, matrix): the significant correlation matrix
    b (array like, matrix): significant inverse correlation matrix

    Returns:
    sig_matrix (array like, matrix): significant correlations
    """
    n_var = len(a)
    sig_matrix = np.zeros((n_var,n_var))
    for i in range(n_var):
        for j in range(n_var):
            if a[i,j]>0 and b[i,j]>0:
                sig_matrix[i,j] = a[i,j]
                # if a[i,j] > 0.35:
                #     sig_matrix[i,j] = a[i,j]
                # else: sig_matrix[i,j] = 0
    return sig_matrix
