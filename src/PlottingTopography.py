import numpy as np
import matplotlib.pyplot as plt

def plot_projections(corr,centers):
    """
    Plots the correlation of each ROIs with the behavior trace on the topography
    Args:
        corr: (array-like, matrix [n x n]): Correlation matrix of data
        centers: (array-like, matrix []) Cell centers in

    Returns:

    """
    fig = plt.figure(figsize=(10, 6))
    if centers.shape[1] == 2:
        plt.scatter(centers[:, 0], centers[:, 1], marker = 'o', s = 10, c = corr, cmap = 'seismic')
        plt.colorbar()
    else:
        ax = fig.add_subplot(111, projection = "3d")
        p = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker = 'o', s = 10, c = corr, cmap = 'seismic')
        fig.colorbar(p)


def plott_(inferred,n_past):
    """
    Plots connectivity matrix inferred into different matrices of corresponding pasts
    Args:
        inferred: (array-like: matrix [# of variables * n_pasts X # of variables]) connectivity matrix inferred
        n_past: (int) number of pasts used in analysis

    Returns: Plotted connectivity matrices corresponding to number of pasts

    """
    nn, n_neur = n_past + 1, inferred.shape[1]
    fig, axs = plt.subplots(1, nn, figsize = (3.5 * nn, 3.5))
    b = 0
    for a in range(nn):
        jj  = inferred[a * n_neur : (a + 1) * n_neur, b * n_neur : (b + 1) * n_neur]
        axs[a].imshow(jj)
        axs[a].axis('off')
    plt.tight_layout()