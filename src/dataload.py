import numpy as np
from pathlib import Path


def load_data(file_path: str) -> np.ndarray:
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



def adj_mtx(n_neur):
    A = np.random.choice([0,0.5,0.85], p=[0.9,0.03,0.07], size=(n_neur,n_neur)) ### A is not the adjacency matrix in the typical sense
    A = 0.5*(A)
    A[0:10,0:10] =np.zeros((10,10))
    for n, i in enumerate(A):
        A[n][n]=1
    for n, i in enumerate(A):
        A[n]=A[n]/np.sum(A[n])
    return A



def continuous_noise_fun(num, l):
    xx = np.linspace(0,500,l)
    noise = np.zeros((num,l))
    for i in range(num):
        a = 2*np.random.normal(0,0.25,size=6)
        c = 500*(np.random.random(size=6))
        s = 1+100*(np.random.random(size=6))
        yy = 0*xx
        for j in range(6):
            yy = yy + a[j]*np.exp(-(xx-c[j])**2/s[j])
        noise[i] = yy
    return noise




