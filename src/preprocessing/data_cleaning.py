"""Data cleaning utilities for calcium imaging traces."""

from __future__ import annotations

import numpy as np


def replace_bad_frames(
    arr: np.ndarray,
    bad_frames: np.ndarray,
    delete_frames: bool,
) -> np.ndarray:
    """Remove or interpolate bad (NaN) frames in an array.

    Parameters
    ----------
    arr : np.ndarray
        Data array of shape ``[n_vars, n_timepoints]`` (variables on rows).
    bad_frames : np.ndarray
        1-D integer array of column indices that are bad.
    delete_frames : bool
        If ``True``, the bad frames are deleted; otherwise they are
        replaced by the linear interpolation of adjacent frames.

    Returns
    -------
    np.ndarray
        Cleaned data without bad frames.
    """
    if delete_frames:
        return np.delete(arr, bad_frames, axis=1)

    arr_copy = arr.copy()
    for var in range(arr_copy.shape[0]):
        for i in bad_frames:
            arr_copy[var, i] = (arr_copy[var, i - 1] + arr_copy[var, i + 1]) / 2.0

    return arr_copy
