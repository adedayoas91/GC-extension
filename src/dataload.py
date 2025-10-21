import numpy as np
import pickle
from pathlib import Path
import scipy


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

    if file_type == '.npy':
        data_array = np.load(file_path)
    elif file_type == '.txt':
        data_array = np.loadtxt(file_path)
    elif file_type == '.pickle':
        with open(file_path, 'rb') as file:
            data_array = pickle.load(file)
    elif file_type == '.mat':
        data_array = scipy.io.loadmat(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return data_array



def simulate_data(A, m, iid=bool):
    """
    Function to simulate a dataset

    Args:
        A: np.ndarray - ground truth connectivity matrix
            Example of A =  np.array([[0,0,0,0,0],
                              [1,0,0,0,0],
                              [0,1,0,0,0],
                              [1,1,0,0,0],
                              [0,0,1,0,0]])

        m: (int) - desired length of the variable
        iid: bool - defines if data to be simulated is time series or iid
            if True; simulates iid data otherwise, time series

    Returns:
        Simulate data: np.ndarray with shape [n_vars, samples]
    """
    np.random.seed(10)

    if iid==True:
        X = np.zeros([A.shape[0],m]).T
        for i, row in enumerate(X):
            for n, var in enumerate(row):
                X[i, n] = np.random.normal(0, 0.1) + np.dot(A[n], X[i])

    else:
        noise = continuous_noise_fun(A.shape[0], m)
        X = np.zeros([A.shape[0], m]).T
        X[0] = np.random.randn(A.shape[0])
        for i, row in enumerate(X[:-1]):
            X[i + 1] = A @ X[i] + 2 * np.random.normal(0,0.25,
                                                   A.shape[0]) + noise[:, i]

    return X.T



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




