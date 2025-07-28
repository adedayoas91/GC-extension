#!/usr/bin/env python3
# coding: utf-8

import numpy as np


class Compute_metrics:
    def __init__(self, conn_mat, A, n_pasts) -> None:
        self.conn_mat = conn_mat
        self.gt = A
        self.n_pasts = n_pasts


    def compute_confusion_matrix(self):
        """
        Compares and returns the comfusion matrix from an inferred connectivity mati=rix and the ground truth matrix
        :param inferred: Connectivity matrix inferred from data
        :param A: Ground truth connectivity matrix
        :return:
        """
        A = self.gt
        inferred = self.conn_mat
        TP_inf = np.sum(np.logical_and(A != 0,inferred!=0))
        FN_inf = np.sum(np.logical_and(A != 0,inferred==0))
        FP_inf = np.sum(np.logical_and(A == 0,inferred!=0))
        TN_inf = np.sum(np.logical_and(A == 0,inferred==0))
        self.confusion_matrix = np.array([[TP_inf,FN_inf],
                                         [FP_inf,TN_inf]])
        return self.confusion_matrix


    def compute_metrics_from_confusion_matrix(self):
        """
        Compute metrixes from the confusion matrix. True and False positives, as well as True and False negatives.
        :return: A vector of Accuracy, Precision, Recall and False Positive Rates
        """

        confusion_matrix_ = self.confusion_matrix.flatten()
        accuracy = (confusion_matrix_[0] + confusion_matrix_[3])/(np.sum(confusion_matrix_))
        precision = confusion_matrix_[0]/(confusion_matrix_[0]+confusion_matrix_[2])
        recall = confusion_matrix_[0]/(confusion_matrix_[0]+confusion_matrix_[1])
        FPR = confusion_matrix_[2]/(confusion_matrix_[2]+confusion_matrix_[3])
        return np.array([accuracy, precision, recall, FPR])


    def repopulate(inf, traces, idx):
        """

        :param inf:
        :param traces:
        :param idx:
        :return:
        """
        inferred_ = np.zeros((traces.shape[0], traces.shape[0]))
        p = np.transpose(np.where(inf != 0))
        for i in range(len(p)):
            inferred_[idx[p[i, 0]], idx[p[i, 1]]] = 1
        return inferred_


    def compute_distance_of_each_projection(self):
        """

        :param centers:
        :param inf:
        :return:
        """

        loc = np.transpose(np.where(inf>0))
        dist = np.zeros(len(loc))
        for i in range(len(loc)):
            p1,p2 = self.centers[loc[i,0]], self.centers[loc[i,1]] 
            dist[i] = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        return dist
        

    def count_number_of_in_out_edges_for_each_ROIs(out_,from_,emitter,reciever):
        """

        :param out_:
        :param from_:
        :param emitter:
        :param reciever:
        :return:
        """
        n_out,n_in = len(out_),len(from_)
        for el in out_:
            if el in emitter:
                n_out -= 1
        for el in from_:
            if el in reciever:
                n_in -= 1
        return n_out, n_in


    def roi_neighbors(self):
        """

        :param inferred_:
        :return:
        """
        inferred_ = self.conn_mat
        nodes_out, nodes_in = {},{}
        for i in range(inferred_.shape[0]):
            nodes_out[i] = np.where(inferred_[i]!=0)[0]
            nodes_in[i] = np.where(inferred_.T[i]!=0)[0]
        return nodes_out,nodes_in


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

### Ploting tools

class Visualize_on_topography(GcStar):
    def __init__(self, n_perm: int, n_pasts: int, n_lags: int) -> None:
        super().__init__(n_perm, n_pasts, n_lags)

        self.topography = None
        self.inf = None  # super().get_connectivity_matrix()
        self.roi_count = None

    def repopulate(self, roi_idx: np.ndarray) -> np.ndarray:
        """
        Repopulates the inferred connectivity matrix obtained from
        identified ROIs selected from a data volume.
        Only necessary for ease of connectivity matrix plotting on
        topography by facilitating ease of pixel coordinate extractions.

        Args:
            pop_count: An integer value stating the number of ROIs in
                the whole population of data where from the volume analysed
                was selected
            roi_idx: A vector of integer indexes of identified ROIs.
            # roi_count: Number of ROIs in the volume the identified
                traces are selected

        Returns:
            Connectivity matrix with the shape of data population.
        """
        inferred_ = np.zeros((self.roi_count, self.roi_count))
        p = np.transpose(np.where(self.inf != 0))
        for i in range(p.shape[0]):
            inferred_[roi_idx[p[i, 0]], roi_idx[p[i, 1]]] = 1  # self.inf[p[i]]

        return inferred_

    def get_cordinates(self, inferred_: np.ndarray):
        """
        A func to make coordinates for each ROIs
        Args:
            inferred_: shape [self.roi_count, self.roi_count]
            topography: the position of ROIs given from data

        Returns:
            coordinates for each ROIs and the edges [t]
        """

        t = np.transpose(np.where(inferred_ > 0))
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
            if self.topography.shape[1] == 3:
                z_val.append([self.topography[t[i, 0], 2],
                              self.topography[t[i, 1], 2]])

        return x_val, y_val, z_val, t

    def plot_conn_mat_on_topography(self,
                                    topography: np.ndarray,
                                    inferred: np.ndarray,
                                    roi_idx: np.ndarray):
        """
        3D visualisation of the connectivity matrix on the topography of fish

        Args:
            topography: 3-dimensional pixels coordinates of ROIs
            roi_idx: A vector of identified ROI indexes

        Returns:
            3D visualisation of neural circuit
        """
        self.topography = topography
        self.inf = inferred
        self.roi_count = topography.shape[1]

        # repopulate the inferred matrix into a bigger matrix
        inferred_rep = self.repopulate(roi_idx=roi_idx)
        # extract coordinates for each ROIs
        x_val, y_val, z_val, t = self.get_cordinates(inferred_rep)
        # visualize in 3-dimensions
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.topography[:, 0], self.topography[:, 1],
                   self.topography[:, 2], color='red', s=10, marker='.')
        ax.scatter(self.topography[roi_idx, 0], self.topography[roi_idx, 1],
                   self.topography[roi_idx, 2], color='green', s=15, marker='*')
        for a in range(len(x_val)):
            ax.plot(x_val[a], y_val[a], z_val[a], lw=0.7, alpha=.7)
        ax.grid(False)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('z-axis')
        return fig

    def plot_correlation_of_ROIs_with_behavior(self,
                                               corr_with_beh: np.ndarray):
        """
        Plots the correlation of each ROIs with the behavior
        trace on the topography

        Args:
            corr_list (array-like, matrix [1 x n_neur]):
             List of correlation coefficient of ind neuron in
             data with the behavior
            centers (np.ndarray): 3D coordinates of all neurons in data

        Returns:

        """
        fig = plt.figure(figsize=(10, 6))
        if self.topography.shape[1] == 2:
            plt.scatter(self.topography[:, 0], self.topography[:, 1],
                        marker='o', s=10, c=corr_with_beh,
                        cmap='seismic')
            plt.colorbar()
        else:
            ax = fig.add_subplot(111, projection="3d")
            p = ax.scatter(self.topography[:, 0], self.topography[:, 1],
                           self.topography[:, 2], marker='o', s=10,
                           c=corr_with_beh, cmap='seismic')
            fig.colorbar(p)


# def single_res(input1,input2):
#     return input1 - (input1.transpose() @ input2) / (input2.transpose() @ input2) * input2
