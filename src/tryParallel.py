#!/usr/bin/env python3.11
# coding: utf-8

import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit, njit
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import mat73
import matplotlib.colors as mcolors
from mpl_toolkits import mplot3d
from tqdm.notebook import tqdm
import concurrent.futures
import logging
import multiprocessing



class GcStar:
    """
    Implementation of Granger causality from causal Bayesian network perspective. 
    Methods:
        fit(self, data: np.ndarray) 
        get_connectivity_matrix(self)
        plot_conn_mat_on_topography(self)
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
        logging.basicConfig(level=logging.INFO)
        self.logger = None  # Initialize logger when needed
        # self.logger = logging.getLogger(__name__)

        self.n_neur = None
        self.n_perm = n_perm
        self.n_pasts = n_pasts
        self.n_lags = n_lags
        self.temporal = temporal

        self.data = None
        self.shifted_data = None
        self.topography = None

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

    def _configure_logging(self, verbose):
        log_level = logging.WARNING  # Default to show only warnings and errors

        if verbose == 1:
            log_level = logging.INFO  # Show informational logs

        elif verbose == 2:
            log_level = logging.DEBUG  # Show detailed debug logs

        logging.basicConfig(level=log_level)


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



    # @jit(nopython=True)
    def perm_test(self, x: np.ndarray, y: np.ndarray) -> float:
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
                x_ = np.roll(x_copy, np.random.randint(30, len(x_copy), 1))

            corr_2 = np.corrcoef(x_, y)[1, 0]
            if np.abs(corr_2) >= np.abs(corr_1):
                count += 1
        return count / self.n_perm


    def get_number_of_neurons(self) -> int:    #private
        """
        Computes the number of neurons/variables from the shape
        of the shifted data obtained from calling
        self.__shift_data() on the original data

        Args:
            trimmed_arr: shifted data from self.__shift_data()

        Returns:
            number of variables/neurons in data
        """
        self.n_neur = self.shifted_data.shape[0] // (self.n_pasts + 1)
        return self.n_neur


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

    def correlation_func(self) -> Tuple[np.ndarray, np.ndarray]:  # private
        """
        Computes unconditional dependence of any variable pair for a range of rows
        using Pearson's correlation as the dependence metric.

        Returns:
            corr[:, :n]: np.ndarray shape [end_row - start_row, self.n_neur]
                Correlation matrix of the data but done on the shifted data for the range.

            pVal_corr: np.ndarray with shape as the correlation matrix.
                Contains the p-values of individual elements in the correlation matrix for the range.
        """
        # Calculate start_row and end_row for this thread
        total_rows = len(self.shifted_data)
        rows_per_thread = total_rows // self.num_corr_threads
        start_row = self.thread_index * rows_per_thread
        end_row = start_row + rows_per_thread if self.thread_index < self.num_corr_threads - 1 else total_rows

        data = self.shifted_data
        corr = np.abs(np.corrcoef(self.shifted_data))
        pVal_corr = np.zeros((end_row - start_row, self.n_neur))

        total_steps = (end_row - start_row) * self.n_neur
        current_step = 0

        for i in range(start_row, end_row):
            for j in range(self.n_neur):
                pVal_corr[i - start_row, j] = self.perm_test(data[i, :], data[j, :])

                current_step += 1
                completion_percentage = (current_step / total_steps) * 100
                self.logger.info(f"Step {current_step}/{total_steps} "
                                 f"({completion_percentage:.2f}% complete)")

        return corr[:, :self.n_neur], pVal_corr

    def inv_correlation_func(self) -> Tuple[np.ndarray, np.ndarray]:  # private
        """
        Computes conditional dependency of any variable pair with Pearson's
        correlation as the dependence metric for a range of rows.

        Returns:
            inv_corr: np.ndarray shape [end_row - start_row, self.n_neur]
                Correlation matrix of the data but done on the shifted data for the range.

            pVal_inv_corr: np.ndarray with shape as the correlation matrix
                contains the p-values of individual elements in the correlation matrix for the range.
        """
        # Calculate start_row and end_row for this thread
        total_rows = len(self.shifted_data)
        rows_per_thread = total_rows // self.num_inv_corr_threads
        start_row = self.thread_index * rows_per_thread
        end_row = start_row + rows_per_thread if self.thread_index < self.num_inv_corr_threads - 1 else total_rows

        data = self.shifted_data
        inv_corr = np.zeros((end_row - start_row, self.n_neur))
        pVal_inv_corr = np.zeros((end_row - start_row, self.n_neur))

        total_steps = (end_row - start_row) * self.n_neur
        current_step = 0

        for i in range(start_row, end_row):
            for j in range(self.n_neur):
                x = data[i]
                y = data[j]
                z = self.get_conditioning_set(i, j)

                x_res = self.__residual(x, z)
                y_res = self.__residual(y, z)

                inv_corr[i - start_row, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                pVal_inv_corr[i - start_row, j] = self.perm_test(x_res, y_res)

                current_step += 1
                completion_percentage = (current_step / total_steps) * 100
                self.logger.info(f"Step {current_step}/{total_steps} "
                                 f"({completion_percentage:.2f}% complete)")

        return inv_corr, pVal_inv_corr

    def fit(self, data: np.ndarray, verbose=1):
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
        # Make a copy of the input data
        self.data = data.copy()

        # Shift the data
        self.shifted_data = self.shift_data(self.data)

        # Set up logging
        self._configure_logging(verbose)
        self.logger = logging.getLogger(__name__)

        # Determine the number of threads for each function
        num_threads = multiprocessing.cpu_count()
        self.num_corr_threads = num_threads // 3
        self.num_inv_corr_threads = num_threads - self.num_corr_threads

        # Create a ThreadPoolExecutor to parallelize the tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit the correlation_func tasks in chunks based on the number of threads allocated
            self.logger.info("Starting correlation_func")
            corr_futures = []
            for i in range(self.num_corr_threads):
                self.thread_index = i
                corr_future = executor.submit(self.correlation_func)
                corr_futures.append(corr_future)

            # Divide the inv_correlation_func work evenly among threads
            self.logger.info("Starting inv_correlation_func")
            inv_corr_futures = []
            for i in range(self.num_inv_corr_threads):
                self.thread_index = i
                inv_corr_future = executor.submit(self.inv_correlation_func)
                inv_corr_futures.append(inv_corr_future)

            # Wait for both tasks to complete and get their results
            self.logger.info("Waiting for correlation_func and inv_correlation_func to complete")
            corr_results = []
            pVal_corr_results = []
            for corr_future in corr_futures:
                corr_chunk, pVal_corr_chunk = corr_future.result()
                corr_results.append(corr_chunk)
                pVal_corr_results.append(pVal_corr_chunk)

            inv_corr_results = []
            pVal_inv_corr_results = []
            for inv_corr_future in inv_corr_futures:
                inv_corr_chunk, pVal_inv_corr_chunk = inv_corr_future.result()
                inv_corr_results.append(inv_corr_chunk)
                pVal_inv_corr_results.append(pVal_inv_corr_chunk)

            corr_ = np.vstack(corr_results)
            pVal_corr_ = np.vstack(pVal_corr_results)
            inv_corr_ = np.vstack(inv_corr_results)
            pVal_inv_corr_ = np.vstack(pVal_inv_corr_results)

        # Assign the results
        self.corr_, self.pVal_corr_ = corr_, pVal_corr_
        self.inv_corr_, self.pVal_inv_corr_ = inv_corr_, pVal_inv_corr_

        return self



    def plot_extended_connectivity_matrix(self, alpha: float = 0.01,
                                        beta: float = 0.001) -> np.ndarray:
        """
        Plots connectivity matrix inferred into different matrices
        of corresponding pasts

        Args:
            inferred: (array-like: matrix [# of variables * n_pasts X number
                        of variables]) connectivity matrix inferred
            n_past: (int) number of pasts used in analysis

        Returns: Plotted connectivity matrices corresponding to
                number of pasts
        """
        nn = self.n_pasts + 1
        fig, axs = plt.subplots(1, nn, figsize=(3.5 * nn, 3.5))
        b = 0
        sig_corr = np.multiply(self.corr_, self.pVal_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.pVal_inv_corr_ <= beta)

        # extended connectivity matrix
        inferred = np.logical_and(sig_corr, sig_inv)

        for a in range(nn):
            jj = inferred[a * self.n_neur:(a + 1) * self.n_neur,
                 b * self.n_neur:(b + 1) * self.n_neur]
            axs[a].imshow(jj)
            axs[a].axis('off')
        plt.tight_layout()


    def get_connectivity_matrix(self, alpha: float = 0.01,
                                        beta: float = 0.001) -> np.ndarray:
        """
        Computes the weighted connectivity matrix from significant
        conditional and unconditional links based on the maximum
        lag allowed for the analysis.
            >If the connections strength are not useful for
            analysis, the connectivity matrix can be binarized

        Args:
            alpha: float, Significance level for unconditional
                    dependence (default value 0.05)

            beta: float, Significance level for conditional
                    dependence (default value 0.001)

        Returns:
            Weighted Connectivity matrix of shape [n_neur, n_neur]
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
            all_.append(inferred[a * n_neur:(a + 1) * n_neur,
                                        b * n_neur:(b + 1) * n_neur])

        self.conn_mat = all_[0]

        for i in range(1, self.n_lags):
            self.conn_mat = np.logical_or(self.conn_mat, all_[i])

        # multiplied with correlation to return connections strength
        self.conn_mat = np.multiply(self.corr_[:n_neur, :], self.conn_mat)
        # remove self connections (diagonal connections)
        np.fill_diagonal(self.conn_mat, 0)
        return self.conn_mat


    ### computing metrics


    def compute_confusion_matrix(self, A: np.ndarray):
        """
        Function to compute the performance of the algorithm.
        Computes the confusion matrix by comparing the ground truth
        to the inferred connectivity matrix.
        Args:
            A: Ground truth connectivity matrix

        Returns:
            Confusion matrix with the form np.array([[TP, FN],
                                                     [FP, TN]])
        """
        TP = np.sum(np.logical_and(A != 0, self.conn_mat != 0))
        FN = np.sum(np.logical_and(A != 0, self.conn_mat == 0))
        FP = np.sum(np.logical_and(A == 0, self.conn_mat != 0))
        TN = np.sum(np.logical_and(A == 0, self.conn_mat == 0))
        self.confusion_matrix = np.array([[TP, FN],
                                          [FP, TN]])

        return self.confusion_matrix



    def compute_metrics(self) -> np.ndarray:
        """
        Computes metrics from the confusion matrix.
        Returns:
            An array containing the computed metrics.
            in the order np.array([accuracy, precision, recall, FPR])
        """
        conf_mat_flatten = self.confusion_matrix.flatten()
        accuracy = ((conf_mat_flatten[0] + conf_mat_flatten[3])
                                    / (np.sum(conf_mat_flatten)))

        precision = conf_mat_flatten[0] / (conf_mat_flatten[0] +
                                                conf_mat_flatten[2])

        recall = conf_mat_flatten[0] / (conf_mat_flatten[0] +
                                                conf_mat_flatten[1])

        FPR = conf_mat_flatten[2] / (conf_mat_flatten[2] +
                                                conf_mat_flatten[3])

        return np.array([accuracy, precision, recall, FPR])



    def get_projection_distances(self, topography):  # REVISE
        """
        Computes the projection distance between ROIs that are linked
        to each other in the connectivity matrix
        Args:
            topography:

        Returns:
            The vector containing the projection distances
        """
        loc = np.transpose(np.where(self.conn_mat > 0))
        dist = np.zeros(len(loc))
        for i in range(len(loc)):
            p1, p2 = topography[loc[i, 0]], topography[loc[i, 1]]
            dist[i] = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
                              + (p2[2] - p1[2]) ** 2)
        return dist

    ### Adding the ploting functions



# visualise
class Visualize_on_topography(GcStar):
    def __init__(self, n_perm: int, n_pasts: int, n_lags: int) -> None:   # , roi_count: int
        super().__init__(n_perm, n_pasts, n_lags)

        self.topography = None
        # self.roi_count = roi_count


    def __repopulate(self, idx: np.ndarray, roi_count: int) -> np.ndarray:
        """
        Repopulates the inferred connectivity matrix obtained from identified
        ROIs selected from a data volume.
        Only necessary for ease of connectivity matrix plotting on
        topography by facilitating ease of pixel coordinate extractions.

        Args:
            pop_count: An integer value stating the number of ROIs in
                the whole population of data where from the volume analysed
                was selected
            idx: A vector of integer indexes of identified ROIs.

        Returns:
            Connectivity matrix with the shape of data population.
        """
        self.inf = super().conn_mat()
        # self.roi_count = roi_count
        inferred_ = np.zeros((self.roi_count, self.roi_count))
        p = np.transpose(np.where(self.inf != 0))
        for i in range(len(p)):
            inferred_[idx[p[i, 0]], idx[p[i, 1]]] = self.inf[p[i]]

        return inferred_


    def __get_cordinates(self):
        """
        A func to make coordinates for each ROIs
        Args:
            inferred: shape [n_neur, n_neur] inferred matrix
            topography: the position of ROIs given from data

        Returns:
            coordinates for each ROIs and the edges [t]
        """
        self.inf = super().conn_mat()
        t = np.transpose(np.where(self.inf > 0))
        point_1 = np.zeros_like(t)
        point_2 = np.zeros_like(t)

        for i in range(len(t)):
            point_1[i] = self.topography[t[i, 0], [0, 1]]
            point_2[i] = self.topography[t[i, 1], [0, 1]]

        x_val = []
        y_val = []
        z_val = []

        for i in range(len(point_1)):
            x_val.append([point_1[i, 0], point_2[i, 0]])
            y_val.append([point_1[i, 1], point_2[i, 1]])
            if self.topography.shape[1] != 3:
                z_val.append([self.topography[t[i, 0], 2],
                                    self.topography[t[i, 1], 2]])

        return x_val, y_val, z_val, t


    def plot_conn_mat_on_topography(self, topography: np.ndarray,
                                                    arr: np.ndarray):
        """
        3D visualisation of the connectivity matrix on the topography of fish

        Args:
            topography: 3-dimensional pixels coordinates of ROIs
            arr: A vector of identified ROI indexes

        Returns:
            3D visualisation of neural circuit
        """
        self.topography = topography
        self.roi_count = self.topography.shape[1]

        x_val, y_val, z_val, t = self.__get_cordinates()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.topography[:, 0], self.topography[:, 1],
                   self.topography[:, 2], color='red', s=10, marker='.')
        ax.scatter(self.topography[arr, 0], self.topography[arr, 1],
                   self.topography[arr, 2], color='green', s=15, marker='*')
        for a in range(len(x_val)):
            ax.plot(x_val[a], y_val[a], z_val[a], lw=0.7, alpha=.7)
        ax.grid(False)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('z-axis')
        return fig



    def plot_correlation_of_ROIs_with_behavior(self, corr_with_beh):
        """
        Plots the correlation of each ROIs with the behavior
        trace on the topography

        Args:
            corr_list: (array-like, matrix [n x n]): Correlation
                        matrix of data
            centers: (array-like, matrix []) Cell centers in

        Returns:

        """
        fig = plt.figure(figsize=(10, 6))
        if self.topography.shape[1] == 2:
            plt.scatter(self.topography[:, 0], self.topography[:, 1],
                        marker = 'o', s = 10, c = corr_with_beh,
                        cmap = 'seismic')
            plt.colorbar()
        else:
            ax = fig.add_subplot(111, projection = "3d")
            p = ax.scatter(self.topography[:, 0], self.topography[:, 1],
                            self.topography[:, 2], marker = 'o', s = 10,
                            c = corr_with_beh, cmap = 'seismic')
            fig.colorbar(p)


