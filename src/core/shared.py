"""Shared numerical utilities used across the Granger causality modules.

All functions here are stateless and contain no class-level state so they
can be imported and composed freely.
"""

from __future__ import annotations

import numpy as np
from numba import jit, njit
from sklearn.linear_model import LinearRegression

from ..constants import PERM_SHIFT_MAX_BUFFER, PERM_SHIFT_MIN


@jit(nopython=True)
def perm_test(x: np.ndarray, y: np.ndarray, n_perm: int) -> float:
    """Compute a permutation p-value for the correlation of two vectors.

    Uses circular shifts (rolling) of *x* to build the null distribution.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays of the same length.
    n_perm : int
        Number of permutations.

    Returns
    -------
    float
        Estimated p-value in [0, 1].
    """
    count = 0
    corr_1 = np.corrcoef(x, y)[1, 0]
    x_copy = x.copy()
    for _ in range(n_perm):
        shift = np.random.randint(PERM_SHIFT_MIN, x.size)
        x_ = np.hstack((x_copy[shift:], x_copy[:shift]))
        corr_2 = np.corrcoef(x_, y)[1, 0]
        if np.abs(corr_2) >= np.abs(corr_1):
            count += 1
    return count / n_perm


@jit(nopython=True)
def perm_test_shift(x: np.ndarray, y: np.ndarray, n_perm: int) -> float:
    """Permutation p-value using rolling shifts (suitable for time-series).

    Parameters
    ----------
    x, y : np.ndarray
        1-D time series of the same length.
    n_perm : int
        Number of permutations.

    Returns
    -------
    float
        Estimated p-value in [0, 1].
    """
    count = 0
    corr_1 = np.corrcoef(x, y)[1, 0]
    shifts = np.random.randint(
        PERM_SHIFT_MIN, len(x) - PERM_SHIFT_MAX_BUFFER, n_perm
    )
    for j in range(n_perm):
        x_ = np.roll(x.copy(), shifts[j])
        corr_2 = np.corrcoef(x_, y)[1, 0]
        if np.abs(corr_2) >= np.abs(corr_1):
            count += 1
    return count / n_perm


@njit
def cross_corr(x: np.ndarray, y: np.ndarray, n_lags: int) -> np.ndarray:
    """Absolute cross-correlation across a symmetric lag window.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays of the same length.
    n_lags : int
        Half-width of the lag window (lags span ``[-n_lags+1, n_lags-1]``).

    Returns
    -------
    np.ndarray
        Absolute cross-correlation values for each lag.
    """
    lags = np.arange(-n_lags + 1, n_lags)
    corr_coef = np.zeros(len(lags))
    for i in range(len(corr_coef)):
        lag = lags[i]
        if lag < 0:
            corr_coef[i] = np.corrcoef(x[: -abs(lag)], y[abs(lag):])[1, 0]
        elif lag == 0:
            corr_coef[i] = np.corrcoef(x, y)[1, 0]
        else:
            corr_coef[i] = np.corrcoef(x[lag:], y[:-lag])[1, 0]
    return np.abs(corr_coef)


def residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Regress the conditioning set *z* out of *x* and return residuals.

    Parameters
    ----------
    x : np.ndarray
        1-D target vector.
    z : np.ndarray
        2-D conditioning set (shape ``[n_vars, n_timepoints]``).

    Returns
    -------
    np.ndarray
        Residual of *x* after removing the linear projection onto *z*.
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(z.T, x)
    return x - np.dot(model.coef_, z) - model.intercept_


def prep_data(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Prepare a lagged data matrix by stacking shifted copies of *x*.

    Parameters
    ----------
    x : np.ndarray
        Data matrix of shape ``[n_vars, n_timepoints]``.
    n_lags : int
        Number of time lags to include (0 returns *x* unchanged).

    Returns
    -------
    np.ndarray
        Stacked lagged matrix of shape
        ``[n_vars * (n_lags + 1), n_timepoints - n_lags]``.
    """
    if n_lags is None or n_lags == 0:
        return x
    x_out = x[:, n_lags:]
    for i in range(n_lags):
        idx1 = n_lags - 1 - i
        idx2 = -i - 1
        x_out = np.r_[x_out, x[:, idx1:idx2]]
    return x_out


def ideal_lp_filter(f_c: float, f_s: float, m: int) -> np.ndarray:
    """Design an ideal low-pass FIR filter kernel in the frequency domain.

    Parameters
    ----------
    f_c : float
        Cut-off frequency in the same units as *f_s*.
    f_s : float
        Sampling frequency.
    m : int
        Filter length (number of taps).

    Returns
    -------
    np.ndarray
        Real-valued impulse response of length *m*.
    """
    amp = np.ones(m)
    cutoff = int(f_c * m / f_s)
    amp[cutoff:-cutoff] = 0
    h_n = np.fft.fftshift(np.real(np.fft.ifft(amp * np.exp(1j * np.zeros(m)))))
    return h_n


def combine(indices: list[int]) -> list[int]:
    """Fill single-element gaps in a sorted list of integer indices.

    Parameters
    ----------
    indices : list[int]
        Sorted list of integer indices.

    Returns
    -------
    list[int]
        Indices with gaps of exactly one filled in.
    """
    result: list[int] = []
    for i in range(len(indices) - 1):
        result.append(indices[i])
        if indices[i] + 2 == indices[i + 1]:
            result.append(indices[i] + 1)
    result.append(indices[-1])
    return result
