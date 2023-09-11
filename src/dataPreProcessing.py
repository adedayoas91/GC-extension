import numpy as np


def replace_nan(arr: np.ndarray, bad_frames: np.ndarray, delete_frames: bool) -> np.ndarray:
    """
    Deletes NAN frames in data and if delete_frames is False, it interpolates to fill the bad frames
    Args:
        arr: # TODO: specify that the array needs to be in the right orientation
        bad_frames:
        delete_frames: (bool) If true, deletes frames and if False, interpolates

    Returns: Data without bad frames
    """
    if delete_frames:
        return np.delete(arr, bad_frames, axis=1)

    arr_copy = arr.copy()
    for var in range(arr_copy.shape[0]):
        for i in bad_frames:
            arr_copy[var, i] = (arr_copy[var, i - 1] + arr_copy[var, i + 1]) / 2

    return arr_copy

