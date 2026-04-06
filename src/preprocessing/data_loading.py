"""Data loading utilities.

Supports NumPy (``.npy``), plain-text (``.txt``), pickle (``.pickle``),
MATLAB (``.mat``) and HDF5-based MATLAB (``.mat`` v7.3) file formats.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.io


def load_data(file_path: str) -> np.ndarray:
    """Load calcium traces from a file.

    Accepted formats: ``.npy``, ``.txt``, ``.pickle``, ``.mat``.

    Parameters
    ----------
    file_path : str
        Path to the data file.

    Returns
    -------
    np.ndarray
        Calcium traces of shape ``[n_rois, n_timepoints]``.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    file = Path(file_path)
    file_type = file.suffix.lower()

    if file_type == ".npy":
        return np.load(file_path)
    elif file_type == ".txt":
        return np.loadtxt(file_path)
    elif file_type == ".pickle":
        with open(file_path, "rb") as fh:
            return pickle.load(fh)
    elif file_type == ".mat":
        return scipy.io.loadmat(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type!r}")


def simulate_data(
    a: np.ndarray, m: int, iid: bool = False
) -> np.ndarray:
    """Simulate a multivariate dataset from a connectivity matrix *a*.

    Parameters
    ----------
    a : np.ndarray
        Ground truth connectivity matrix of shape ``[n_vars, n_vars]``.
    m : int
        Number of time-steps (or i.i.d. samples) to generate.
    iid : bool
        If ``True``, generates i.i.d. observations; otherwise simulates
        a time series via a linear auto-regressive process.

    Returns
    -------
    np.ndarray
        Simulated data of shape ``[n_vars, m]``.
    """
    np.random.seed(10)

    if iid:
        x = np.zeros((m, a.shape[0]))
        for i, row in enumerate(x):
            for n in range(len(row)):
                x[i, n] = np.random.normal(0, 0.1) + np.dot(a[n], x[i])
    else:
        noise = _continuous_noise(a.shape[0], m)
        x = np.zeros((m, a.shape[0]))
        x[0] = np.random.randn(a.shape[0])
        for i in range(m - 1):
            x[i + 1] = (
                a @ x[i]
                + 2 * np.random.normal(0, 0.25, a.shape[0])
                + noise[:, i]
            )

    return x.T


def adj_matrix(n_vars: int) -> np.ndarray:
    """Generate a random sparse adjacency matrix.

    Parameters
    ----------
    n_vars : int
        Number of variables / nodes.

    Returns
    -------
    np.ndarray
        Row-normalised adjacency matrix of shape ``[n_vars, n_vars]``
        with a forced zero block on the first 10 × 10 sub-matrix.
    """
    a = np.random.choice(
        [0, 0.5, 0.85], p=[0.9, 0.03, 0.07], size=(n_vars, n_vars)
    )
    a = 0.5 * a
    if n_vars >= 10:
        a[:10, :10] = np.zeros((10, 10))
    for n in range(a.shape[0]):
        a[n, n] = 1.0
    for n in range(a.shape[0]):
        row_sum = np.sum(a[n])
        if row_sum > 0:
            a[n] = a[n] / row_sum
    return a


def _continuous_noise(n_vars: int, length: int) -> np.ndarray:
    """Generate structured continuous noise for simulation.

    Parameters
    ----------
    n_vars : int
        Number of noise traces to generate.
    length : int
        Length of each trace.

    Returns
    -------
    np.ndarray
        Noise matrix of shape ``[n_vars, length]``.
    """
    xx = np.linspace(0, 500, length)
    noise = np.zeros((n_vars, length))
    for i in range(n_vars):
        a = 2 * np.random.normal(0, 0.25, size=6)
        c = 500 * np.random.random(size=6)
        s = 1 + 100 * np.random.random(size=6)
        yy = np.zeros_like(xx)
        for j in range(6):
            yy = yy + a[j] * np.exp(-((xx - c[j]) ** 2) / s[j])
        noise[i] = yy
    return noise
