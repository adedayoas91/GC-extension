"""Core Granger causality implementation.

The :class:`causalisedGrangerCausality` class provides the primary interface
for fitting conditional and unconditional Granger causality models to
multivariate time-series data.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    RISING_SHIFT_MAX_BUFFER,
    RISING_SHIFT_MIN,
)
from .shared import ideal_lp_filter, perm_test, residual

try:
    from cdt.metrics import SHD

    _CDT_AVAILABLE = True
except ImportError:
    _CDT_AVAILABLE = False

logger = logging.getLogger(__name__)


class causalisedGrangerCausality:
    """Granger causality from a causal Bayesian network perspective.

    Parameters
    ----------
    n_perm : int
        Number of permutations used in p-value computations.
    n_pasts : int
        Number of past states (time lags) to include.
    n_lags : int
        Maximum lag to consider in connectivity.
    temporal : bool
        ``True`` if data are a time series; ``False`` for i.i.d. data.
    method : str
        Conditioning strategy.  ``"cgc"`` for conventional Granger
        causality; ``"fcgc"`` for the full-conditioning variant.
    """

    corr_: Optional[np.ndarray]
    p_val_corr_: Optional[np.ndarray]
    inv_corr_: Optional[np.ndarray]
    p_val_inv_corr_: Optional[np.ndarray]

    def __init__(
        self,
        n_perm: int,
        n_pasts: int,
        n_lags: int,
        temporal: bool,
        method: str,
    ) -> None:
        self.n_perm = n_perm
        self.n_pasts = n_pasts
        self.n_lags = n_lags
        self.temporal = temporal
        self.method = method

        self.n_neur: Optional[int] = None
        self.f_s: Optional[float] = None
        self.f_c: Optional[float] = None
        self.m: Optional[int] = None
        self.data: Optional[np.ndarray] = None
        self.shifted_data: Optional[np.ndarray] = None
        self.topography: Optional[np.ndarray] = None
        self.conn_mat: Optional[np.ndarray] = None
        self.confusion_matrix: Optional[np.ndarray] = None

        self.corr_ = None
        self.p_val_corr_ = None
        self.inv_corr_ = None
        self.p_val_inv_corr_ = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_time_series(self) -> bool:
        """``True`` if the data were treated as a time series."""
        return self.temporal

    @property
    def number_of_lags(self) -> int:
        """Maximum lag used in the analysis."""
        return self.n_lags

    @property
    def number_of_perms(self) -> int:
        """Number of permutations used for p-value estimation."""
        return self.n_perm

    @property
    def number_of_pasts(self) -> int:
        """Number of past states included in the analysis."""
        return self.n_pasts

    @property
    def number_of_neurons(self) -> int:
        """Number of variables derived from the shape of ``shifted_data``."""
        if self.shifted_data is None:
            raise ValueError("Call fit() before accessing number_of_neurons.")
        self.n_neur = self.shifted_data.shape[0] // (self.n_pasts + 1)
        return self.n_neur

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _configure_logging(self, verbose: int) -> None:
        level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        logging.basicConfig(
            level=level_map.get(verbose, logging.WARNING), force=True
        )

    # ------------------------------------------------------------------
    # Data manipulation
    # ------------------------------------------------------------------

    def shift_data(self, arr: np.ndarray) -> np.ndarray:
        """Stack time-lagged copies of *arr* to create the design matrix.

        Parameters
        ----------
        arr : np.ndarray
            Shape ``[n_vars, n_timepoints]``.

        Returns
        -------
        np.ndarray
            Shape ``[n_vars * (n_pasts + 1), n_timepoints - n_pasts]``.
        """
        self.n_neur = arr.shape[0]
        if self.n_pasts == 0:
            return arr
        trimmed = arr[:, self.n_pasts:]
        for i in range(self.n_pasts):
            idx1 = self.n_pasts - 1 - i
            idx2 = -i - 1
            trimmed = np.r_[trimmed, arr[:, idx1:idx2]]
        return trimmed

    # ------------------------------------------------------------------
    # Correlation / conditioning
    # ------------------------------------------------------------------

    def correlation_func(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute unconditional Pearson correlation and permutation p-values.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[n_vars, n_timepoints]``.

        Returns
        -------
        corr : np.ndarray
            Shape ``[n_shifted, n_vars]``, absolute correlation values.
        p_val_corr : np.ndarray
            Corresponding permutation p-values.
        """
        x_copy = x.copy()
        self.n_neur = x_copy.shape[0]
        data = self.shift_data(x_copy)
        n = data.shape[0]
        corr = np.abs(np.corrcoef(self.shifted_data))
        p_val_corr = np.zeros((n, self.n_neur))

        total = n * self.n_neur
        step = 0
        for i in range(n):
            for j in range(self.n_neur):
                p_val_corr[i, j] = perm_test(data[i, :], data[j, :], self.n_perm)
                step += 1
                logger.info(
                    "corr step %d/%d (%.1f%%)", step, total, 100 * step / total
                )
        return corr[:, : self.n_neur], p_val_corr

    def get_conditioning_set(
        self, x: np.ndarray, i: int, j: int
    ) -> np.ndarray:
        """Extract the conditioning set for the pair *(i, j)*.

        Parameters
        ----------
        x : np.ndarray
            Original (un-shifted) data.
        i : int
            Row index of the putative cause in the shifted matrix.
        j : int
            Row index of the effect in the shifted matrix.

        Returns
        -------
        np.ndarray
            Conditioning-set rows.
        """
        self.shifted_data = self.shift_data(x.copy())
        i_ = i % self.n_neur
        x_idx = [i_ + a * self.n_neur for a in range(i // self.n_neur)]
        z = np.delete(
            self.shifted_data,
            np.r_[np.array(x_idx, dtype=int), [i, j]],
            axis=0,
        )
        return z

    def inv_correlation_func(
        self, x: np.ndarray, method: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute conditional correlation and permutation p-values.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[n_vars, n_timepoints]``.
        method : str
            ``"fcgc"`` or ``"cgc"``.

        Returns
        -------
        inv_corr : np.ndarray
        p_val_inv_corr : np.ndarray
        """
        self.n_neur = x.copy().shape[0]
        data = self.shift_data(x.copy())
        n = data.shape[0]
        inv_corr = np.zeros((n, self.n_neur))
        p_val_inv_corr = np.zeros((n, self.n_neur))

        total = n * self.n_neur
        step = 0
        for i in range(n):
            for j in range(self.n_neur):
                xi = data[i]
                yi = data[j]
                self.data = x.copy()
                if method == "fcgc":
                    z = np.delete(data.copy(), [i, j], axis=0)
                else:
                    z = self.get_conditioning_set(self.data, i, j)
                x_res = residual(xi, z)
                y_res = residual(yi, z)
                inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                p_val_inv_corr[i, j] = perm_test(x_res, y_res, self.n_perm)
                step += 1
                logger.info(
                    "inv step %d/%d (%.1f%%)", step, total, 100 * step / total
                )
        return inv_corr, p_val_inv_corr

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray, verbose: int = 1) -> "causalisedGrangerCausality":
        """Fit the Granger causality model to data.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[n_vars, n_timepoints]``.
        verbose : int
            ``0`` = warnings only; ``1`` = info; ``2`` = debug.

        Returns
        -------
        causalisedGrangerCausality
            The fitted instance (enables method chaining).
        """
        self.data = x.copy()
        self.shifted_data = self.shift_data(self.data)
        self._configure_logging(verbose)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            logger.info("Fitting: starting correlation_func")
            corr_future = executor.submit(self.correlation_func, x)
            logger.info("Fitting: starting inv_correlation_func")
            inv_future = executor.submit(
                self.inv_correlation_func, x, method=self.method
            )
            corr_, p_val_corr_ = corr_future.result()
            inv_corr_, p_val_inv_corr_ = inv_future.result()

        self.corr_ = corr_
        self.p_val_corr_ = p_val_corr_
        self.inv_corr_ = inv_corr_
        self.p_val_inv_corr_ = p_val_inv_corr_
        return self

    # ------------------------------------------------------------------
    # Connectivity matrix
    # ------------------------------------------------------------------

    def get_connectivity_matrix(
        self,
        simulation: bool,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> np.ndarray:
        """Compute the weighted connectivity matrix.

        Parameters
        ----------
        simulation : bool
            ``True`` when data are simulated with a known ground truth.
        alpha : float
            Significance level for unconditional dependence.
        beta : float
            Significance level for conditional dependence.

        Returns
        -------
        np.ndarray
            Weighted connectivity matrix, shape ``[n_vars, n_vars]``.
        """
        sig_corr = np.multiply(self.corr_, self.p_val_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.p_val_inv_corr_ <= beta)
        inferred = np.logical_and(sig_corr, sig_inv)
        n_neur = inferred.shape[1]
        slices = [
            inferred[a * n_neur: (a + 1) * n_neur, :n_neur]
            for a in range(self.n_pasts + 1)
        ]

        if simulation:
            self.conn_mat = slices[1]
        else:
            self.conn_mat = (
                np.logical_or(slices[0], slices[1])
                if self.n_lags == 1
                else slices[0]
            )

        for i in range(1, self.n_lags + 1):
            self.conn_mat = np.logical_or(self.conn_mat, slices[i])

        self.conn_mat = np.multiply(self.corr_[:n_neur, :], self.conn_mat)
        return self.conn_mat

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_confusion_matrix(
        self, a: np.ndarray, simulation: bool
    ) -> np.ndarray:
        """Compare ``conn_mat`` against ground truth *a*.

        Parameters
        ----------
        a : np.ndarray
            Ground truth adjacency matrix.
        simulation : bool
            If ``True``, transposes *a* before comparison.

        Returns
        -------
        np.ndarray
            ``[[TP, FP], [FN, TN]]``.
        """
        if simulation:
            a = a.T
        tp = int(np.sum(np.logical_and(a != 0, self.conn_mat != 0)))
        fn = int(np.sum(np.logical_and(a != 0, self.conn_mat == 0)))
        fp = int(np.sum(np.logical_and(a == 0, self.conn_mat != 0)))
        tn = int(np.sum(np.logical_and(a == 0, self.conn_mat == 0)))
        self.confusion_matrix = np.array([[tp, fp], [fn, tn]])
        return self.confusion_matrix

    def all_metrics(self) -> np.ndarray:
        """Compute classification metrics from ``confusion_matrix``.

        Returns
        -------
        np.ndarray
            ``[accuracy, precision, recall, fpr, balanced_accuracy, f1]``.
        """
        cm = self.confusion_matrix.flatten()
        tp, fp, fn, tn = int(cm[0]), int(cm[1]), int(cm[2]), int(cm[3])
        accuracy = (tp + tn) / float(np.sum(cm))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (specificity + recall) / 2
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return np.array([accuracy, precision, recall, fpr, balanced_accuracy, f1])

    def compute_shd_sid(
        self, a: np.ndarray, inf: np.ndarray, simulation: bool
    ) -> int:
        """Compute Structural Hamming Distance (SHD).

        Requires the optional ``cdt`` package.

        Parameters
        ----------
        a : np.ndarray
            Ground truth matrix.
        inf : np.ndarray
            Inferred matrix.
        simulation : bool
            If ``True``, transposes *a*.

        Returns
        -------
        int
            SHD value.

        Raises
        ------
        ImportError
            If ``cdt`` is not installed.
        """
        if not _CDT_AVAILABLE:
            raise ImportError(
                "The 'cdt' package is required for compute_shd_sid. "
                "Install it with: pip install cdt"
            )
        if simulation:
            a = a.T
        self.shd_ = SHD(target=a, pred=inf, double_for_anticausal=False)
        return self.shd_

    # ------------------------------------------------------------------
    # Rising flanks helpers (kept for API parity with original GcStar)
    # ------------------------------------------------------------------

    def _perm_test_rising(self, x: np.ndarray, y: np.ndarray) -> float:
        """Permutation p-value using small shifts (for rising-flank arrays)."""
        count = 0
        corr_1 = np.corrcoef(x, y)[1, 0]
        shifts = np.random.randint(
            RISING_SHIFT_MIN, len(x) - RISING_SHIFT_MAX_BUFFER, self.n_perm
        )
        for j in range(self.n_perm):
            x_ = np.roll(x.copy(), shifts[j])
            corr_2 = np.corrcoef(x_, y)[1, 0]
            if np.abs(corr_2) >= np.abs(corr_1):
                count += 1
        return count / self.n_perm

    def lp_filter(self) -> np.ndarray:
        """Return the ideal low-pass filter kernel using stored parameters."""
        return ideal_lp_filter(self.f_c, self.f_s, self.m)

    def get_rising_flanks(
        self, arr: np.ndarray, f_c: float, f_s: float, m: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract rising-flank samples from *arr* via low-pass filtering.

        Parameters
        ----------
        arr : np.ndarray
            1-D time series.
        f_c : float
            Cut-off frequency.
        f_s : float
            Sampling frequency.
        m : int
            Filter length.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(arr[idx], idx)`` where *idx* marks rising-flank positions.
        """
        self.f_c = f_c
        self.f_s = f_s
        self.m = m
        h_n = self.lp_filter()
        conv = np.convolve(arr, h_n, "same")
        j = np.diff(conv) > np.mean(np.diff(conv) > 0.5) / 10
        idx = np.where(j > 0)[0]
        return arr[idx], idx

    def fit_rising(
        self, x: np.ndarray, idx: np.ndarray, verbose: int = 1
    ) -> "causalisedGrangerCausality":
        """Fit Granger causality on rising-flank segments.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[n_vars, n_timepoints]``.
        idx : np.ndarray
            Per-variable rising-flank index arrays.
        verbose : int
            Verbosity level.

        Returns
        -------
        causalisedGrangerCausality
            The fitted instance.
        """
        self.n_neur = x.shape[0]
        n_idx = len(idx)
        corr = np.zeros((n_idx * self.n_pasts, n_idx))
        p_val_corr = np.zeros_like(corr)
        inv_corr = np.zeros_like(corr)
        p_val_inv_corr = np.zeros_like(corr)

        for i in tqdm(range(n_idx * self.n_pasts)):
            for j in range(n_idx):
                i_ = i if i < x.shape[0] else i % n_idx
                same_idx = sorted(set(idx[i_]).intersection(idx[j]))
                if len(same_idx) > 0.25 * min(len(idx[i_]), len(idx[j])):
                    dat = self.shift_data(x[:, same_idx])
                    corr[i, j] = np.abs(np.corrcoef(dat[i], dat[j])[1, 0])
                    p_val_corr[i, j] = self._perm_test_rising(dat[i], dat[j])
                    xi = dat[i]
                    yi = dat[j]
                    self.data = x.copy()[:, same_idx]
                    z = self.get_conditioning_set(self.data, i, j)
                    x_res = residual(xi, z)
                    y_res = residual(yi, z)
                    inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                    p_val_inv_corr[i, j] = self._perm_test_rising(x_res, y_res)
                else:
                    corr[i, j] = 0.0
                    p_val_corr[i, j] = 1.0
                    inv_corr[i, j] = 0.0
                    p_val_inv_corr[i, j] = 1.0

        self.corr_ = corr
        self.p_val_corr_ = p_val_corr
        self.inv_corr_ = inv_corr
        self.p_val_inv_corr_ = p_val_inv_corr
        return self

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_extended_connectivity_matrix(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        """Plot the connectivity matrices for each past-state lag.

        Parameters
        ----------
        alpha : float
            Unconditional significance threshold.
        beta : float
            Conditional significance threshold.
        """
        nn = self.n_pasts + 1
        _fig, axs = plt.subplots(1, nn, figsize=(3.5 * nn, 3.5))
        sig_corr = np.multiply(self.corr_, self.p_val_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.p_val_inv_corr_ <= beta)
        inferred = np.logical_and(sig_corr, sig_inv)
        for a in range(nn):
            block = inferred[
                a * self.n_neur: (a + 1) * self.n_neur,
                : self.n_neur,
            ]
            axs[a].imshow(block)
            axs[a].axis("off")
        plt.tight_layout()


def compare_with_gt(
    a: np.ndarray, inf: np.ndarray, simulation: bool
) -> np.ndarray:
    """Colour-code the confusion between ground truth *a* and inferred *inf*.

    Encoding:  TP → 40, FP → 30, FN → 20, TN → 10.

    Parameters
    ----------
    a : np.ndarray
        Ground truth adjacency matrix.
    inf : np.ndarray
        Inferred connectivity matrix.
    simulation : bool
        If ``True``, transposes *a* before comparison.

    Returns
    -------
    np.ndarray
        Integer matrix with the colour codes above.
    """
    if simulation:
        a = a.T
    return (
        40 * np.logical_and(a != 0, inf != 0)
        + 30 * np.logical_and(a == 0, inf != 0)
        + 20 * np.logical_and(a != 0, inf == 0)
        + 10 * np.logical_and(a == 0, inf == 0)
    )
