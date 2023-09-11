# functions to get conditioning set
import numpy as np


def get_past(X: np.ndarray, n_past: int) -> np.ndarray:
    """
    :params:
    * `X`: np.ndarray of shape (num_vars, num_timesteps)
    * `n_past`: int, number of timelags to consider

    :returns:
    * np.ndarray of shape (n_past+1, num_vars, num_timesteps-n_past) with lags
        increasing along the first axis.
    """
    assert len(X.shape) == 2, \
        "X must be a 2-dimensional array"
    if n_past == 0:
        return X.copy().reshape(1, *X.shape)
    past_matrices = []
    for j in range(n_past + 1):
        X_past_j = X[:, n_past - j:X.shape[1] - j]
        past_matrices.append(X_past_j)
    X_past = np.stack(past_matrices)
    return X_past


def get_conditioning_set(
        X: np.ndarray,
        n_past: int,
        i_lag: int,
        i_ind: int,
        j_ind: int
) -> np.ndarray:
    """
    :params:
    * `i_lag`: int, the number of timesteps back that we consider the
        independent variable at.
    * `i_ind`: the index (maximum X.shape[0]) of the independent variable.
    * `j_ind`: int, the index of the dependent variable.

    All indices in range(0, X.shape[0]) that are not `i_ind` are considered to
    be indices of latent variables.

    :returns:
    * np.ndarray of 2 dimensions, where each row represents a conditioned
        variable, and each column includes the historical values of said
        variable.
    """
    num_vars = X.shape[0]
    X_past = get_past(X, n_past)
    # get the latent variable indices
    all_indices = np.arange(num_vars)
    ij_mask = np.isin(all_indices, [i_ind, j_ind])
    z_indices = all_indices[~ij_mask]  # everything that isn't i or j is z
    # `X_past` has shape (n_past, num_vars, X.shape[1]-n_past)

    # From the independent variable, we want to return everything before but
    # not including the "current" value at `i_lag`
    i_past = X_past[i_lag + 1:, [i_ind], :]
    # i_past shape (history up to i_lag, 1, X.shape[1]-n_past)

    # For the latent variable, we want to return everything up to and at the
    # sampe time as the independent variable
    z_past = X_past[i_lag:, z_indices, :]
    # z_past shape (history up to i_lag+1, X.shape[0]-2, X.shape[1]-n_past)

    # For the dependent variable, we return all times in the past but not the
    # current value
    j_past = X_past[1:, [j_ind], :]
    # j_past shape (history up to current time, 1, X.shape[1]-n_past)

    # reshape everything to be compatible shape
    i_past_reshaped = i_past.reshape(-1, i_past.shape[-1])
    j_past_reshaped = j_past.reshape(-1, j_past.shape[-1])
    z_past_reshaped = z_past.reshape(-1, z_past.shape[-1])
    # stack it back into a matrix and return
    return np.vstack([i_past_reshaped, j_past_reshaped, z_past_reshaped])


def index_converter(X: np.ndarray, i: int, j: int):
    """Helper function to convert inputs `i` and `j` to lags and indices."""
    j_ind = j
    i_ind = i % X.shape[0]
    i_lag = i // X.shape[0]
    return i_lag, i_ind, j_ind


if __name__ == "__main__":
    X = np.arange(60).reshape(3,-1)

    print("Setting i=12, j=2")
    i = 12
    j = 2
    print("translating variables to i_lag, i_ind, j_ind...")
    i_lag, i_ind, j_ind = index_converter(X, i, j)
    # j_ind = j
    # i_ind = i % X.shape[0]
    # i_lag = i // X.shape[0]
    print("i_lag:", i_lag, "i_ind:", i_ind, "j_ind:", j_ind)
    print("Resulting conditioning set:")
    print(get_conditioning_set(X=X, n_past=5, i_lag=i_lag, i_ind=i_ind, j_ind=j_ind))

    ## =============================== ##

    print("Setting i=2, j=0")
    i = 4
    j = 0
    print("translating variables to i_lag, i_ind, j_ind...")
    i_lag, i_ind, j_ind = index_converter(X, i, j)
    # j_ind = j
    # i_ind = i % X.shape[0]
    # i_lag = i // X.shape[0]
    print("i_lag:", i_lag, "i_ind:", i_ind, "j_ind:", j_ind)
    print("Resulting conditioning set:")
    print(get_conditioning_set(X=X, n_past=5, i_lag=i_lag, i_ind=i_ind, j_ind=j_ind))
