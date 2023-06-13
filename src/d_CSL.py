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


###################################################################
## Analysis
###################################################################


class Gc_star():
    def __init__(self, n_perm, n_pasts, temporal):
        self.number_of_perm = n_perm
        self.number_of_lags = n_pasts
        self.temporal = True

    def check_data_is_time_sereis(self):
        return self.temporal

    def get_number_of_lags(self):
        return self.number_of_lags
    
    def load_pickle_data(self):
        raise NotImplementedError

    def get_number_of_perm(self):
        return self.number_of_perm

    def dependence_test_method(self, dependence_test_choice):
        self.dependence_test = dependence_test_choice

    def get_data_path(self, file_path):
        return file_path
    
    def get_bad_frames(self, bad_frame_times):
        return bad_frame_times 


    def load_data(self, file_path):
        """
        File name should be given in string with the path to where it is located.
        :param file_name: the name of file containing data. 
        :return: traces (loaded data)
        """
        self.traces = np.load(file_path)
        return self.traces



    @njit
    def cross_corr(self, x, y):
        """Returns the absolute values of cross correlation of 2 variables with any specified lag window

        Parameters:
        x (array like): array of a variable of any length
        y (array like): array of a variable of same length as x
        n_lags (int): lag window size (function takes both positive and negative lag of the same window)

        Returns:
        cross correlation of 2 variable (array like), each entry of the array, the correlationat each lag window.

    """
        n_lags = self.number_of_pasts 
        lags = np.arange(-n_lags+1, n_lags)
        corr_coef = np.zeros(len(lags))
        for i in range(len(corr_coef)):
            if lags[i]< 0:
                corr_coef[i] = np.corrcoef(x[:-np.abs(lags[i])], y[np.abs(lags[i]):])[1, 0]
            elif lags[i]==0:
                corr_coef[i] = np.corrcoef(x, y)[1,0]
            else:
                corr_coef[i] = np.corrcoef(x[lags[i]:], y[:-lags[i]])[1,0]

        return np.abs(corr_coef)



    def cross_correlation(self):
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

        data, n_perm, n_lags = self.traces, self.number_of_perm, self.number_of_pasts
        corr_coef = np.abs(np.corrcoef(data))
        pVal_corr, pVal_Xcorr = np.zeros_like(corr_coef),np.zeros_like(corr_coef)
        
        ##########################################################
        # computing cross correlation of all neuron with lag of 20
        # selected the max correlations at which ever lag it occurs
        ##########################################################
        n_var,max_ccoef_mat,max_corr_lag = corr_coef.shape[0],np.zeros_like(corr_coef),np.zeros_like(corr_coef)
        lags = np.arange(-n_lags+1,n_lags).astype(int)
        for n in range(0,n_var):
            for i in range(n,n_var):
                pVal_corr[n,i] = self.perm_test(data[n,:],data[i,:],n_perm)
                pVal_corr[i,n] = pVal_corr[n,i]
                ccor = self.cross_corr(data[n,:], data[i,:],n_lags)
                max_ccoef_mat[n,i] = np.max(ccor)
                max_ccoef_mat[i,n] = max_ccoef_mat[n,i]
                
                max_corr_lag[n,i] = lags[np.argmax(ccor)]
                max_corr_lag[i,n]= max_corr_lag[n,i]
                
                ## Computing p_Values of cross correlation 
                l = lags[np.argmax(ccor)]
                if l < 0:
                    pVal_Xcorr[n,i] = self.perm_test(data[n,:-np.abs(l)],data[i,np.abs(l):],n_perm)
                elif l==0:
                    pVal_Xcorr[n,i] = self.perm_test(data[n,:],data[i,:],n_perm)
                else:
                    pVal_Xcorr[n,i] = self.perm_test(data[n,l:],data[i,:-l],n_perm)
                pVal_Xcorr[i,n] = pVal_Xcorr[n,i]
        max_corr_lag = max_corr_lag.astype(int)

        return corr_coef,pVal_corr,max_ccoef_mat,max_corr_lag,pVal_Xcorr



    @jit(nopython=True)
    def perm_test(self, x, y):
        """
        Computes p_value of correlation of two iid generated variables.
        Args:
            x: (array like, vector): Realisation of a variable x
            y: (array like, vector): Realisation of a variable y
            shuffle: Number of permutations
        Returns: p_value
        """
        shuffle = self.number_of_perm
        count, corr_1 = 0,np.corrcoef(x,y)[1,0]
        for j in range(self.number_of_perm):
            if not self.temporal: 
                x_copy = np.copy(x)
                np.random.shuffle(x_copy)
            else:
                x_ = np.roll(x_copy, np.random.randint(30, len(x)), 1)
            corr_2 = np.corrcoef(x_copy,y)[1,0]
            if np.abs(corr_2) >= np.abs(corr_1):
                count+=1
        return count/shuffle


    def residual(self, x, z):
        """
        Computes the residuals of a variable x by regressing a conditioning set z out of it
        :param x: Variable in question to regress out the conditioning set 
        :param z: Conditioning set 
        :return:
        """
        model = LinearRegression(fit_intercept=True)
        model.fit(z.T, x)
        coefs, intercept = model.coef_, model.intercept_
        return x - np.dot(coefs, z) - intercept
        # return residual


        
    def correlation_func(self, traces):   # naame compute_dependence_with_corelation()
        n_perm, n_past = self.number_of_perm, self.number_of_pasts
        data = self.prep_data(traces, n_past)
        corr, n, n_neur = np.abs(np.corrcoef(data)), traces.shape[0], data.shape[0]
        pVal_corr = np.zeros((n_neur, n))
        for i in range(n_neur):
            for j in range(n):
                pVal_corr[i, j] = self.perm_test(data[i, :], data[j, :], n_perm)
        return corr[:, :n], pVal_corr


    def inv_correlation_func(self, traces):  # compute_conditional_dependence_with_corelation()
        n_perm, n_past = self.number_of_perm, self.number_of_pasts
        data = self.prep_data(traces, n_past)
        n_neur = traces.shape[0]
        inv_corr, pVal_inv_corr = np.zeros((data.shape[0], n_neur)), np.zeros((data.shape[0], n_neur))
        for i in range(0, data.shape[0]):
            for j in range(0, n_neur):
                x, y, z = data[i], data[j], np.delete(data, [i, j], axis = 0)
                x_res, y_res = self.residual(x, z), self.residual(y, z)
                inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1,0])
                pVal_inv_corr[i, j] = self.perm_test(x_res, y_res, n_perm)
        return inv_corr, pVal_inv_corr



    def conditioning_set(self, traces, i, j):
        """
        Identiffies the varriables in the conditioning set for granger causality implementation.
        Args:
            traces: (array-like, matrices shape [# of variables X samples]) Data
            n_perm: (int) number of permutations for p_value computations
            n_past: (int) number of pasts desired
        Returns: correlation, correspondomg p-Values, inverse correlation and the p-valeus
        """
        X, n_past = traces, self.number_of_pasts
        n = X.shape[0]
        k, data = i//n, self.prep_data(X, n_past)
        z_ = np.delete(data, np.r_[np.arange(k * n), [i]], axis=0)
        y_ = self.prep_data(X[j, :].reshape((1, X.shape[1])), n_past)
        z = np.r_[z_, y_[1:k]]
        return z



    #implementing GC
    def GC_with_CBN(self, traces):
        """
        Performs the analysis
        Args:
            traces: (array-like, matrices shape [# of variables X samples]) Data
            n_perm: (int) number of permutations for p_value computations
            n_past: (int) number of pasts desired
        Returns: correlation, correspondomg p-Values, inverse correlation and the p-valeus
        """
        n_perm, n_past = self.number_of_perm, self.number_of_pasts
        data = self.prep_data(traces, n_past)
        n, n_neur = traces.shape[0], data.shape[0]
        corr,pVal_corr = self.correlation_func(traces,n_perm,n_past)
        inv_corr,pVal_inv_corr =np.zeros((n_neur,n)), np.zeros((n_neur, n))
        for i in range(0, n_neur):
            for j in range(0, n):
                x, y, z = data[i],data[j], self.conditioning_set(traces, i, j)             
                x_res,y_res = self.residual(x, z), self.residual(y, z)
                inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                pVal_inv_corr[i, j] = self.perm_test(x_res, y_res, n_perm)
        self.corr = corr 
        self.corr = corr
        return corr,pVal_corr, inv_corr,pVal_inv_corr


    def connectivity_matrix_(self,alpha,beta):
        """

        :param corr:
        :param pVal_corr:
        :param inv_corr:
        :param pVal_inv_corr:
        :param alpha:
        :param beta:
        :param n_past:
        :return:
        """
        corr, pVal_corr = self.corr, self.pVal_corr
        inv_corr, pVal_inv_corr = self.inv_corr, self.pVal_inv_corr
        n_past = self.number_of_pasts

        # determine significance with chosen alpha and beta
        sig_corr = np.multiply(corr, pVal_corr <= alpha)        # compute significant correlation matrix
        sig_inv = np.multiply(inv_corr, pVal_inv_corr <= beta)  # compute significant partial correlation matrix
        
        return np.logical_and(sig_corr, sig_inv)          # inferred matrix
        
        
        def split_inferred_(self, inferred): 
            # self.plott_(inferred, n_past)   
            b, all_, n_neur = 0, [], inferred.shape[1]
            n_past = self.number_of_pasts
            # merging results for the GC order used
            for a in range(n_past+1):     
                all_.append(inferred[a*n_neur:(a+1)*n_neur,b*n_neur:(b+1)*n_neur])
            nn, new_inf = n_past+1, all_[0]
            for i in range(1,n_past):        # use 'nn' if all matrices are to be used, here i take out the last one
                new_inf = np.logical_or(new_inf,all_[i])
        #     new_inf = np.multiply(corr[:n_neur,:],new_inf)   # multiplied with correlation to determine the strength of connections
        #     np.fill_diagonal(new_inf,0)      # self connectivity removed
            plt.figure()
            plt.imshow(new_inf,vmin=0,vmax=1)
        #     plt.colorbar()
            return new_inf # take into account variations in lags  


    def connectivity_matrix_from_markovianity_check(corr,pVal_corr,inv_corr,pVal_inv_corr,alpha,beta,n_past):
        """
        Only select matrix at 1 past
        :param corr:
        :param pVal_corr:
        :param inv_corr:
        :param pVal_inv_corr:
        :param alpha:
        :param beta:
        :param n_past:
        :return:
        """
        sig_corr = np.multiply(corr,pVal_corr<=alpha)        # compute significant correlation matrix
        sig_inv = np.multiply(inv_corr,pVal_inv_corr<=beta)  # compute significant partial correlation matrix
        inferred = np.logical_and(sig_corr,sig_inv)          # inferred matrix
    #     plott_(inferred,n_past)   
        b, all_, n_neur = 0, [], inferred.shape[1]
        # merging results for the GC order used
        for a in range(n_past+1):     
            all_.append(inferred[a*n_neur:(a+1)*n_neur,b*n_neur:(b+1)*n_neur])
        new_inf = all_[1]
    #     np.fill_diagonal(new_inf,0)      # self connectivity removed
    #     plt.figure()
    #     plt.imshow(new_inf,vmin=0,vmax=1)
        return new_inf  # only lag speified


    def proceed(self):
        """

        :param traces:
        :param n_perm:
        :param n_past:
        :return:
        """
        traces, n_perm, n_past = self.traces, self.number_of_perm, self.number_of_pasts
        corr,pVal_corr = self.correlation_func(traces,n_perm,n_past)    # no conditioning on the past
        inv_corr,pVal_inv_corr = self.inv_correlation_func(traces,n_perm,n_past)
        return corr, pVal_corr, inv_corr, pVal_inv_corr



    


    def modify_inv_corr(self, inv_corr, a):
        # a is the percentile to be discarded
        inv_corr = self.inv_corr
        m = (inv_corr.max() - inv_corr.min()) * a
        return np.multiply(inv_corr >= m, inv_corr)



    ##################################################################
    ##################################################################
    # with mutual information
    ##################################################################
    ##################################################################

    # @jit(nopython=True)
    # def perm_test_MI(x,y,N):
    #     count,mi = 0,computeMI(x,y)
    #     val = np.random.randint(30, len(x) - 30, N)
    #     for j in range(N):
    #         x_copy = np.copy(x)
    #         x_ = np.roll(x_copy, val[j])
    #         mi_ = computeMI(x_, y)
    #         if mi_ >= mi:
    #             count+=1
    #     return count/N



    # @jit(nopython=True)
    # def computeMI(self, x, y):
    #     sum_mi = 0.0
    #     x_value_list,y_value_list = np.unique(x),np.unique(y)
    #     Px = np.array([len(x[x==xval])/float(len(x)) for xval in x_value_list])
    #     Py = np.array([len(y[y==yval])/float(len(y)) for yval in y_value_list])
    #     for i in range(len(x_value_list)):
    #         if Px[i] ==0.:
    #             continue
    #         sy = y[x == x_value_list[i]]
    #         if len(sy)== 0:
    #             continue
    #         pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
    #         t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
    #         sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    #     return sum_mi



    # def MI_func(self):
    #     traces,n_perm,n_past = self.traces, self.number_of_perm, self.number_of_pasts
    #     data = prep_data(traces,n_past)
    #     n,n_neur = traces.shape[0],data.shape[0]
    #     MI,pVal_MI = np.zeros((n_neur,n)), np.zeros((n_neur,n))
    #     for i in range(n_neur):
    #         for j in range(n):
    #             MI[i,j],pVal_MI[i,j] = computeMI(data[i,:],data[j,:]), perm_test_MI(data[i,:],data[j,:],n_perm)
    #     return MI[:,:n], pVal_MI



    # def analysis_GC_MI(self):
    #     traces,n_perm,n_past = self.traces, self.number_of_perm, self.number_of_pasts
    #     data = self.prep_data(traces,n_past)
    #     n, n_neur = traces.shape[0], data.shape[0]
    #     MI,pVal_MI = MI_func(traces,n_perm,n_past)
    #     CMI,pVal_CMI =np.zeros((n_neur,n)),np.zeros((n_neur,n))
    #     for i in tqdm(range(0,n_neur)):
    #         for j in range(0,n):
    #             x,y = data[i,:], data[j,:],
    #             z = conditioning_set(traces,n_past,i,j)
    #             x_res,y_res = residual(x,z),residual(y,z)
    #             CMI[i,j],pVal_CMI[i,j] = computeMI(x_res,y_res),perm_test_MI(x_res,y_res,n_perm)
    #     return MI,pVal_MI,CMI,pVal_CMI


##################################################################
##################################################################