import numpy as np
import matplotlib.pyplot as plt


class Visualize_on_topography():
    def __init__(self, connectivity_matrix, cell_centers,n_pasts) -> None:
        self.conn_mat = connectivity_matrix
        self.cell_centers = cell_centers
        self.number_of_pasts = n_pasts

    def extract_coordinates(self):
        """
        Extracts 3D coordinates of each ROIs
        :param inferred: shape [n x n] inferred matrix
        :param topography: the position of ROIs given from data
        :return: coordinates for each ROIs and the edges [t]
        """
        inferred, topography = self.conn_mat, self.cell_centers
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

        return x_val,y_val,z_val, t



    def plot_correlation_of_ROIs_with_behavior(self, corr_list):
        """
        Plots the correlation of each ROIs with the behavior trace on the topography
        Args:
            corr_list: (array-like, matrix [n x n]): Correlation matrix of data
            centers: (array-like, matrix []) Cell centers in

        Returns:

        """
        centers = self.cell_centers
        fig = plt.figure(figsize=(10, 6))
        if centers.shape[1] == 2:
            plt.scatter(centers[:, 0], centers[:, 1], marker = 'o', s = 10, c = corr_list, cmap = 'seismic')
            plt.colorbar()
        else:
            ax = fig.add_subplot(111, projection = "3d")
            p = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker = 'o', s = 10, c = corr_list, cmap = 'seismic')
            fig.colorbar(p)


    def plot_connectivity_matrix_at_all_lags(self):
        """
        Plots connectivity matrix inferred into different matrices of corresponding pasts
        Args:
            inferred: (array-like: matrix [# of variables * n_pasts X # of variables]) connectivity matrix inferred
            n_past: (int) number of pasts used in analysis

        Returns: Plotted connectivity matrices corresponding to number of pasts

        """
        inferred, n_past = self.conn_mat, self.number_of_pasts 
        nn, n_neur = n_past + 1, inferred.shape[1]
        fig, axs = plt.subplots(1, nn, figsize = (3.5 * nn, 3.5))
        b = 0
        for a in range(nn):
            jj  = inferred[a * n_neur : (a + 1) * n_neur, b * n_neur : (b + 1) * n_neur]
            axs[a].imshow(jj)
            axs[a].axis('off')
        plt.tight_layout()



    def plot_connectivity_matrix_of_identified_ROIs_from_volume(inferred,centers,arr):
        """

        :param inferred:
        :param centers:
        :param arr:
        :return:
        """
        x_val,y_val,z_val = self.extract_coordinates(inferred,centers)
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