"""ICA-based denoising pipeline for calcium imaging data.

:class:`ICADecomposition` wraps ``sklearn.decomposition.FastICA`` with
automatic component-count selection via eigenvalue decomposition and
provides spectral visualisation helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – side-effect import
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA


class ICADecomposition:
    """ICA-based denoising pipeline.

    Parameters
    ----------
    max_iter : int
        Maximum number of FastICA iterations.
    tolerance : float
        Convergence tolerance for FastICA.
    n_comps : int
        Number of independent components (overridden when ``eig_dec=True``
        is passed to :meth:`fit`).
    f_c : float
        Cut-off frequency used when building the low-pass filter kernel.
    f_s : float
        Sampling frequency.
    """

    def __init__(
        self,
        max_iter: int,
        tolerance: float,
        n_comps: int,
        f_c: float,
        f_s: float,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_comps = n_comps
        self.f_c = f_c
        self.f_s = f_s

        self.data: Optional[np.ndarray] = None
        self.ic_comps: Optional[np.ndarray] = None
        self.ics: Optional[np.ndarray] = None
        self.mixing_mat: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Component-count selection
    # ------------------------------------------------------------------

    def get_n_comps_with_eig_dec(
        self, data: np.ndarray, var_to_keep: float
    ) -> int:
        """Select the number of components that capture *var_to_keep* variance.

        Plots the eigenvalue scree and returns the minimal component count
        that explains at least *var_to_keep* fraction of total variance.

        Parameters
        ----------
        data : np.ndarray
            Input data (shape ``[n_vars, n_timepoints]``).
        var_to_keep : float
            Variance fraction to retain (e.g. ``0.95``).

        Returns
        -------
        int
            Number of components.
        """
        self.data = data.copy()
        cov = np.cov(self.data)
        eig_values, _ = np.linalg.eig(cov)
        total_var = np.sum(eig_values)
        cumvar, i = 0.0, 1
        while cumvar < var_to_keep:
            cumvar = float(np.sum(eig_values[:i]) / total_var)
            i += 1
        plt.stem(np.arange(len(eig_values)), eig_values)
        plt.vlines(
            i, 0, eig_values.max(),
            label=f"{i} eig_vals = {np.sum(eig_values[:i]) / total_var:.3f}",
            color="r",
        )
        plt.legend()
        plt.title("Eigenvalues")
        return i

    # ------------------------------------------------------------------
    # Filter helper
    # ------------------------------------------------------------------

    def ideal_lp(self, m: int) -> np.ndarray:
        """Return an ideal low-pass FIR filter kernel of length *m*."""
        amp = np.ones(m)
        cutoff = int(self.f_c * m / self.f_s)
        amp[cutoff:-cutoff] = 0
        h_n = np.fft.fftshift(np.real(np.fft.ifft(amp * np.exp(1j * np.zeros(m)))))
        return h_n

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self, data: np.ndarray, var_to_keep: float, eig_dec: bool
    ) -> "ICADecomposition":
        """Decompose *data* into independent components.

        Parameters
        ----------
        data : np.ndarray
            Calcium traces of shape ``[n_vars, n_timepoints]``.
        var_to_keep : float
            Variance fraction for automatic component selection (only
            used when ``eig_dec=True``).
        eig_dec : bool
            If ``True``, selects the number of components automatically
            via eigenvalue decomposition; otherwise uses ``self.n_comps``.

        Returns
        -------
        ICADecomposition
            The fitted instance (enables method chaining).
        """
        self.data = data.copy()
        if eig_dec:
            self.n_comps = self.get_n_comps_with_eig_dec(data, var_to_keep)
        else:
            self.n_comps = self.data.shape[0]

        ica = FastICA(
            n_components=self.n_comps,
            tol=self.tolerance,
            max_iter=self.max_iter,
            whiten="unit-variance",
        )
        self.ic_comps = ica.fit_transform(self.data.T)
        self.mixing_mat = ica.mixing_
        self.mean = ica.mean_

        self.ics = np.zeros((self.n_comps, self.data.shape[1]))
        for i in range(self.n_comps):
            self.ics[i, :] = np.abs(np.fft.fft(self.ic_comps[:, i]))

        return self

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_ics(self) -> None:
        """Plot each independent component as a time trace."""
        n_comps = self.ic_comps.shape[1]
        fig, axs = plt.subplots(n_comps, 1, figsize=(15, 1.3 * n_comps))
        for i in range(n_comps):
            axs[i].vlines(
                np.arange(30, 1210, 60),
                ymin=self.ic_comps.T[i, :].min(),
                ymax=self.ic_comps.T[i, :].max(),
                ls="--",
                color="g",
                lw=0.6,
            )
            axs[i].plot(self.ic_comps.T[i, :], label=str(i), lw=0.6)
            axs[i].set_ylim([-0.2, 0.2])
            axs[i].legend()

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_ics(
        self, n_clus: int, f_s: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster ICs by spectral content using KMeans.

        Parameters
        ----------
        n_clus : int
            Number of clusters.
        f_s : float
            Sampling frequency for Welch estimation.

        Returns
        -------
        new_mat : np.ndarray
            PCA 3-D representation of each IC.
        predictions : np.ndarray
            Cluster assignment for each IC.
        """
        ic_welch = np.zeros((self.ics.shape[1], 126))
        for i in range(self.ics.shape[1]):
            _, ic_welch[i, :] = welch(
                self.ics[:, i],
                f_s,
                return_onesided=True,
                nperseg=250,
                noverlap=125,
            )
        kmeans = KMeans(n_clusters=n_clus)
        predictions = kmeans.fit_predict(ic_welch[:, 30:])
        pca = PCA(n_components=3)
        new_mat = pca.fit_transform(ic_welch[:, 30:])
        return new_mat, predictions

    @staticmethod
    def plot_clusters(new_mat: np.ndarray, predictions: np.ndarray) -> None:
        """Scatter plot of clusters in 3-D PCA space.

        Parameters
        ----------
        new_mat : np.ndarray
            PCA-reduced data (from :meth:`cluster_ics`).
        predictions : np.ndarray
            Cluster labels.
        """
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(int(predictions.max()) + 1):
            ax.scatter(
                new_mat[:, 0][predictions == i],
                new_mat[:, 1][predictions == i],
                new_mat[:, 2][predictions == i],
                label=str(i),
                cmap="brg",
            )
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.legend()

    def plot_ft_spectrals(self, f_s: float, n_comps: int) -> None:
        """Plot FFT amplitude and Welch power spectra for each IC.

        Parameters
        ----------
        f_s : float
            Sampling frequency.
        n_comps : int
            Number of components to plot.
        """
        _fig, axs = plt.subplots(n_comps, 2, figsize=(15, 1.2 * n_comps))
        freqs = np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, self.ics.shape[0]))
        for i in range(n_comps):
            axs[i, 0].plot(freqs, np.abs(np.fft.fft(self.ics[:, i])))
            axs[i, 0].set_title(f"FFT Amplitude Spectrum of IC {i}")
            axs[i, 0].grid()
            axs[i, 0].set_xlim([0, f_s / 2])
            for overlap in [100, 125, 150]:
                f, pxx = welch(
                    self.ics[:, i],
                    f_s,
                    return_onesided=True,
                    nperseg=250,
                    noverlap=overlap,
                )
                axs[i, 1].plot(f, pxx, label=f"overlap={overlap}", alpha=0.75)
            axs[i, 1].set_title(f"Welch spectrum {i}")
            axs[i, 1].grid()
            axs[i, 1].legend()
        plt.tight_layout()

    def plot_spectrals(
        self, n_clus: int, cluster_assignments: np.ndarray, f_s: float
    ) -> None:
        """Plot overlapping spectra per cluster group.

        Parameters
        ----------
        n_clus : int
            Number of clusters.
        cluster_assignments : np.ndarray
            2-D array where column index 3 holds the cluster label.
        f_s : float
            Sampling frequency.
        """
        _fig, axs = plt.subplots(1, n_clus, figsize=(15, 3))
        freqs = np.fft.fftshift(
            np.linspace(-f_s / 2, f_s / 2, self.ics.shape[1])
        )
        for i in range(n_clus):
            group = self.ics[np.where(cluster_assignments[:, 3] == i)]
            for j in range(group.shape[0]):
                axs[i].plot(freqs, group[j, :])
            axs[i].plot(freqs, group.mean(0), color="black", lw=2, label="mean")
            axs[i].set_xlim([0, 3])
            axs[i].set_title(f"cluster {i} with {group.shape[0]} entries")
            axs[i].legend()

    def plot_log_spectrals(
        self, n_clus: int, cluster_assignments: np.ndarray, f_s: float
    ) -> None:
        """Plot log-power spectra per cluster group.

        Parameters
        ----------
        n_clus : int
            Number of clusters.
        cluster_assignments : np.ndarray
            2-D array where column index 3 holds the cluster label.
        f_s : float
            Sampling frequency.
        """
        _fig, axs = plt.subplots(1, n_clus + 1, figsize=(15, 3))
        freqs = np.fft.fftshift(
            np.linspace(-f_s / 2, f_s / 2, self.ics.shape[1])
        )
        for i in range(n_clus):
            group = self.ics[np.where(cluster_assignments[:, 3] == i)]
            for j in range(group.shape[0]):
                axs[i].plot(freqs, np.log(group[j, :]))
            axs[i].plot(
                freqs, np.log(group.mean(0)), color="black", lw=1, label="mean log"
            )
            axs[i].set_xlim([0, 0.75])
            axs[i].set_title(f"LogPower clus {i}, {group.shape[0]}")
            axs[i].legend()
            axs[n_clus].plot(
                freqs, np.log(group.mean(0)), lw=1, label=f"clus {i}"
            )
        axs[n_clus].set_xlim([0, 0.75])
        axs[n_clus].set_ylim([-3, 3])
        axs[n_clus].legend()
