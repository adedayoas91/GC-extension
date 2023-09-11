import numpy as np
from pathlib import Path


def load(file_path: str) -> np.ndarray:
    """
    Loads calcium traces from a file. Accepted file types:
     * npy
     * txt
     * pickle
    Args:
        file_path: string, file path of the file which will be loaded

    Returns:
        np.array of calcium tracers of the shape [n_ROIs x Time]
    """
    file = Path(file_path)
    file_type = file.suffix

    if file_type == 'npy':
        data_array = np.load(file_path)
    elif file_type == 'txt':
        data_array = np.loadtxt(file_path)
    else:
        raise NotImplementedError
    return data_array

