"""Rising-flank variant of the Granger causality model.

:class:`RisingFlankGrangerCausality` restricts the analysis to the
rising flanks of calcium imaging transients to improve signal-to-noise.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from ..constants import RISING_SHIFT_MAX_BUFFER, RISING_SHIFT_MIN
from .shared import ideal_lp_filter

logger = logging.getLogger(__name__)


class RisingFlankGrangerCausality:
    """Granger causality computed on rising-flank segments.

    Parameters
    ----------
    n_perm : int
        Number of permutations for p-value estimation.
    n_pasts : int
        Number of past time-lags.
    n_lags : int
        Maximum lag for connectivity.
    f_s : float
        Sampling frequency (Hz).
    seg_len : int
        Minimum segment length to retain.
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
        f_s: float,
        seg_len: int,
    ) -> None:
        self.n_perm = n_perm
        self.n_pasts = n_pasts
        self.n_lags = n_lags
        self.seg_len = seg_len
        self.fs = f_s

        self.n_neur: Optional[int] = None
        self.data: Optional[np.ndarray] = None
        self.shifted_data: Optional[np.ndarray] = None
        self.topography: Optional[np.ndarray] = None

        self.corr_ = None
        self.p_val_corr_ = None
        self.inv_corr_ = None
        self.p_val_inv_corr_ = None

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def moving_average(self, data: np.ndarray, win_size: int) -> list[float]:
        """Compute a simple moving average.

        Parameters
        ----------
        data : np.ndarray
            1-D array.
        win_size : int
            Window size.

        Returns
        -------
        list[float]
            Moving-average values (length ``len(data) - win_size + 1``).
        """
        self.data = data.copy()
        avg = []
        for i in range(len(data) - win_size + 1):
            avg.append(round(float(np.mean(data[i: i + win_size])), 4))
        return avg

    def ideal_lp(self, f_c: float, m: int) -> np.ndarray:
        """Return an ideal low-pass FIR filter kernel.

        Parameters
        ----------
        f_c : float
            Cut-off frequency.
        m : int
            Filter length (number of taps).

        Returns
        -------
        np.ndarray
            Real-valued impulse response of length *m*.
        """
        return ideal_lp_filter(f_c, self.fs, m)

    # ------------------------------------------------------------------
    # Data shifting / conditioning
    # ------------------------------------------------------------------

    def shift_data(self, arr: np.ndarray) -> np.ndarray:
        """Stack time-lagged copies of *arr*.

        Parameters
        ----------
        arr : np.ndarray
            Shape ``[n_vars, n_timepoints]``.

        Returns
        -------
        np.ndarray
            Shape ``[n_vars * (n_pasts + 1), n_timepoints - n_pasts]``.
        """
        self.data = arr.copy()
        self.n_neur = self.data.shape[0]
        if self.n_pasts == 0:
            return arr
        trimmed = arr[:, self.n_pasts:]
        for i in range(self.n_pasts):
            idx1 = self.n_pasts - 1 - i
            idx2 = -i - 1
            trimmed = np.r_[trimmed, arr[:, idx1:idx2]]
        self.shifted_data = trimmed
        return trimmed

    def get_past(self, x: np.ndarray) -> np.ndarray:
        """Return stacked past matrices for all lags up to ``n_pasts``.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[n_vars, n_timepoints]``.

        Returns
        -------
        np.ndarray
            Shape ``[n_pasts + 1, n_vars, n_timepoints - n_pasts]``.
        """
        if x.ndim != 2:
            raise ValueError("x must be 2-dimensional.")
        if self.n_pasts == 0:
            return x.copy().reshape(1, *x.shape)
        pasts = [
            x[:, self.n_pasts - j: x.shape[1] - j]
            for j in range(self.n_pasts + 1)
        ]
        return np.stack(pasts)

    def get_conditioning_set(self, i: int, j: int) -> np.ndarray:
        """Build the conditioning set for the pair *(i, j)*.

        Parameters
        ----------
        i : int
            Row index of the putative cause in the shifted matrix.
        j : int
            Row index of the effect.

        Returns
        -------
        np.ndarray
            Stacked conditioning variables.
        """
        x = self.data.copy()
        num_vars = self.n_neur
        j_ind = j
        i_ind = i % num_vars
        i_lag = i // num_vars

        x_past = self.get_past(x)
        all_indices = np.arange(num_vars)
        z_indices = all_indices[~np.isin(all_indices, [i_ind, j_ind])]

        i_past = x_past[i_lag + 1:, [i_ind], :].reshape(-1, x_past.shape[-1])
        j_past = x_past[1:, [j_ind], :].reshape(-1, x_past.shape[-1])
        z_past = x_past[i_lag:, z_indices, :].reshape(-1, x_past.shape[-1])
        return np.vstack([i_past, j_past, z_past])

    # ------------------------------------------------------------------
    # Private numerical helpers
    # ------------------------------------------------------------------

    def _perm_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Permutation p-value for the rising-flank context (short arrays)."""
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

    @staticmethod
    def _residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Regress *z* out of *x* and return residuals."""
        model = LinearRegression(fit_intercept=True)
        model.fit(z.T, x)
        return x - np.dot(model.coef_, z) - model.intercept_

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------

    def fit_rising(
        self, x: np.ndarray, idx: np.ndarray, verbose: int = 1
    ) -> "RisingFlankGrangerCausality":
        """Fit the rising-flank Granger causality model.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[n_vars, n_timepoints]``.
        idx : np.ndarray
            Per-variable rising-flank index arrays.
        verbose : int
            Verbosity (unused; kept for API parity).

        Returns
        -------
        RisingFlankGrangerCausality
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
                if len(same_idx) > 0.1 * min(len(idx[i_]), len(idx[j])):
                    dat = self.shift_data(x[:, same_idx])
                    corr[i, j] = np.abs(np.corrcoef(dat[i], dat[j])[1, 0])
                    p_val_corr[i, j] = self._perm_test(dat[i], dat[j])
                    xi = dat[i]
                    yi = dat[j]
                    self.data = x.copy()[:, same_idx]
                    z = self.get_conditioning_set(i, j)
                    x_res = self._residual(xi, z)
                    y_res = self._residual(yi, z)
                    inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                    p_val_inv_corr[i, j] = self._perm_test(x_res, y_res)
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


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------


def cut_rising_flanks_lp(
    arr: np.ndarray, f_c: float, f_s: float, m: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rising-flank samples via low-pass filtering.

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
        ``(arr[idx], idx)`` where *idx* marks rising positions.
    """
    h_n = ideal_lp_filter(f_c, f_s, m)
    conv = np.convolve(arr, h_n, "same")
    j = np.diff(conv) > np.mean(np.diff(conv) > 0.5) / 10
    idx = np.where(j > 0)[0]
    return arr[idx], idx


def group_consecutive_indices(row: np.ndarray) -> list[list[int]]:
    """Group consecutive integers in *row* into sublists.

    Parameters
    ----------
    row : np.ndarray
        1-D array of integer indices.

    Returns
    -------
    list[list[int]]
        Sublists of consecutive index runs.
    """
    row_list = list(row)
    groups: list[list[int]] = []
    temp: list[int] = []
    for i in range(len(row_list) - 1):
        temp.append(row_list[i])
        if row_list[i] + 1 != row_list[i + 1]:
            groups.append(temp.copy())
            temp = []
    temp.append(row_list[-1])
    groups.append(temp)
    return groups


def matrix_difference(inf_rising: np.ndarray, inf: np.ndarray) -> np.ndarray:
    """Return elements of *inf_rising* that differ from *inf*.

    Parameters
    ----------
    inf_rising : np.ndarray
    inf : np.ndarray

    Returns
    -------
    np.ndarray
        Copy of *inf_rising* with matching positions set to ``False``.
    """
    mat = inf_rising.copy()
    mat[inf_rising == inf] = False
    return mat
