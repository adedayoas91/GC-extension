#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
# from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy.signal import welch
import pandas as pd
# from new_utils import *
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


class ICA_dec:
    """
    Implementation of proposed denoising pipeline
    Methods:
        fit(self, data: np.ndarray)
        get_connectivity_matrix(self)
    """
    def __init__(self, max_iter: int, tolerance: float, n_comps: int):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_comps = n_comps

        self.data = None

        self.ic_comps = None
        self.ICs = None
        self.mixing_mat = None
        self.mean = None

    def __get_n_comps_with_eig_dec(self, var_to_keep: float) -> int:
        """
        Decompose data covariance matrix to compute the eigen values.
        The number of eigen values that expresses 95% of the data variance.
        The number can then be used as the number of components to fit ICA.
        Args:
            var_to_keep: The percentage of variance to keep
        Returns:
            The number of eigen values that make up var_to_keep
        """
        cov = np.cov(self.data)
        eig_values, eig_vectors = np.linalg.eig(cov)
        var = 0
        i = 1
        while var < var_to_keep:
            var = np.sum(eig_values[:i]) / np.sum(eig_values)
            i += 1
        plt.stem(np.arange(len(eig_values)), eig_values)
        plt.vlines(i, 0, eig_values.max(),
                   label='{} eig_vals = {}'.format(i, np.sum(eig_values[:i])
                                                / np.sum(eig_values)), color='r')
        plt.legend()
        plt.title('Eigen values')
        return i

    def fit(self, data, eig_dec: bool, var_to_keep=None):
        """

        Args:
            data: Loaded calcium transcient with the ROIs on the rows
            i.e., [n_neur, Time]
            eig_dec: Defines whether user wants to compute the number of eigen
                values that make up at least var_to_keep percent of variance.

            var_to_keep:

        Returns:

        """
        self.data = data.copy()
        if eig_dec == True:
            self.n_comps = self.__get_n_comps_with_eig_dec(self, var_to_keep)
        else:
            self.n_comps = self.data.shape[0]
        ica = FastICA(n_components = self.n_comps, tol = self.tolerance,
                            max_iter = self.max_iter, whiten = 'unit-variance')
        self.ic_comps = ica.fit_transform(self.data.T)
        self.mixing_mat = ica.mixing_
        self.mean = ica.mean_
        self.ICs = np.zeros((self.n_comps, self.data.shape[1]))
        for i in range(self.n_comps):
            self.ICs[i, :] = np.abs(np.fft.fft(self.ic_comps[:, i]))


    # visualise
    def plott_ics(self):
        fig, ax = plt.subplots(self.ic_comps.shape[1], 1,
                                figsize=(15, 1.3 * self.ic_comps.shape[1]))
        for i in range(self.ic_comps.shape[1]):
            ax[i].vlines(np.arange(30, 1210, 60),
                        ymin=self.ic_comps.T[i, :].min(),
                        ymax=self.ic_comps.T[i, :].max(), ls='--',
                        color='g', lw=.6)
            ax[i].plot((self.ic_comps.T[i, :]), label='{}'.format(i), lw=.6)
            ax[i].set_ylim([-.2, .2])
            ax[i].legend()


    # clusterigs
    def __cluster_ICs(self, n_clus, f_s):
        """
        ICs (array ): The matrix of all ICs; shape [n_features, n_ICs].
        l (int): the length to which mat is truncated to cluster on

        Args:
            n_clus: (int) number of cluster (2 or 3)
            f_s: sampling frequency of data

        Returns:
            new_mat: (np.ndarray) PCA representation of each ICs in 3D
            predictions: Cluster assignment of each ICs
        """
        # Kmeans on truncated spectra
        IC_welch = np.zeros((self.ICs.shape[1], 126))
        for i in range(self.ICs.shape[1]):
            _, IC_welch[i, :] = welch(self.ICs[:, i], f_s,
                                        return_onesided = True,
                                        nperseg = 250, noverlap = 125)
        kmeans = KMeans(n_clusters=n_clus)
        predictions = kmeans.fit_predict(IC_welch[:, 30:])

        # apply PCA to visualise clustering in 3 dimensions
        pca = PCA(n_components=3)
        new_mat = pca.fit_transform(IC_welch[:, 30:])
        c = kmeans.cluster_centers_
        centers = pca.transform(c)
        return new_mat, predictions


    # plottings
    def plot_FT_spectrals(self, f_s, n_comps):
        """

        Args:
            Ics: all ICA returns indep components with returned
                    shape [n_features,n_var]
            f_s: sampling frequency
            n_comps is the number of ICs

        f_s:
        n_comps:

        Returns:

        """
        fig, ax = plt.subplots(n_comps, 2, figsize=(15, 1.2 * n_comps))
        for i in range(n_comps):
            ax[i, 0].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2,
                                        self.ICs.shape[0])),
                                        np.abs(np.fft.fft(self.ICs[:, i])))
            ax[i, 0].set_title('FFT Amplitude Spectrum of IC {}'.format(i))
            ax[i, 0].grid()
            ax[i, 0].set_xlim([0, f_s / 2])
            J = [100, 125, 150]
            for j in J:
                f, Pxx_den = welch(self.ICs[:, i], f_s, return_onesided=True,
                                                    nperseg = 250, noverlap = j)
                ax[i, 1].plot(f, Pxx_den, label='overlap = {}'.format(j),
                                                                alpha=.75)
                ax[i, 1].set_title('Welch spectrum {}'.format(i))
                ax[i, 1].grid()
                ax[i, 1].legend()
        plt.tight_layout()


    def plot_clusters(new_mat, predictions):  # ,centers
        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        for i in range(int(predictions.max()) + 1):
            ax.scatter(new_mat[:, 0][predictions == i],
                       new_mat[:, 1][predictions == i],
                       new_mat[:, 2][predictions == i],
                       label=i, cmap='brg')

        # ax.scatter([0,0],centers[0,1],centers[0,2],c='b',s=80,label='Noise')
        # ax.scatter(centers[1,0],centers[1,1],centers[1,2],c='g',s=80,label='Signal')
        # ax.scatter(centers[2,0],centers[2,1],centers[2,2],c='g',s=80,label='Undecided')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        ax.legend()


    def plottings_spectrals(self, n_clus, alll, f_s):
        """
        All spectrals overlapped in each group
        Args:
            n_clus: (int) number of clusters
            alll:
            f_s:

        Returns:

        """

        fig, ax = plt.subplots(1, n_clus, figsize=(15, 3))
        for i in range(n_clus):
            group = self.ICs[np.where(alll[:, 3] == i)]
            for j in range(group.shape[0]):
                ax[i].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2,
                                           self.ICs.shape[1])), group[j, :])
            ax[i].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2,
                                       self.ICs.shape[1])), group.mean(0),
                                       color = 'black', lw = 2, label = 'mean')
            ax[i].set_xlim([0, 3])
            ax[i].set_title('cluster {} with {} enteries'.format(i,
                                                            group.shape[0]))
            ax[i].legend()



    def plottings_group_spectrals(self, n_clus, alll, f_s):
        fig, ax = plt.subplots(1, n_clus, figsize=(10, 3))
        for i in range(n_clus):
            group = self.ICs[np.where(alll[:, 4] == i)]
            for j in range(group.shape[0]):
                ax[i].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2,
                                            self.ICs.shape[1])), group[j, :])
            ax[i].set_xlim([0, 3])
            ax[i].set_title('cl {} with {}'.format(i, group.shape[0]))



    def plottings_logSpectral(self, n_clus, alll, f_s):
        """
        Plots the log of all spectrals in each group
        Args:
            n_clus:
            alll:
            f_s: (int) Sampling frequency
        """
        fig, ax = plt.subplots(1, n_clus + 1, figsize=(15, 3))
        for i in range(n_clus):
            group = self.ICs[np.where(alll[:, 3] == i)]
            for j in range(group.shape[0]):
                ax[i].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2,
                                        self.ICs.shape[1])), np.log(group[j, :]))
            ax[i].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2,
                                    self.ICs.shape[1])), np.log(group.mean(0)),
                                            color='black', lw=1, label='mean log')
            ax[i].set_xlim([0, .75])
            # ax[i].set_ylim([-5, 2])
            ax[i].set_title('LogPower clus {}, {}'.format(i, group.shape[0]))
            ax[i].legend()

            ax[n_clus].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2,
                                self.ICs.shape[1])), np.log(group.mean(0)),
                                        lw = 1, label = 'clus {}'.format(i))
            ax[n_clus].set_xlim([0, .75])
            ax[n_clus].set_ylim([-3, 3])
            ax[n_clus].legend()


    # proceed function without returning the individuals results
def compute(traces, n_clus, f_s):
    a = eig_dec(traces)
    ic_comps, IC_ft, A, mean = ica_dec(traces, a, t=0.0001, max_=500)
    new_mat, predictions = cluster(IC_ft, n_clus, f_s)
    al = np.c_[new_mat.round(1), predictions.round(1)]
    plot_clusters(new_mat, predictions)
    plottings_spectrals(n_clus, al)
    plottings_logSpectral(n_clus, al)

if __name__ == "__main__":
    compute()

