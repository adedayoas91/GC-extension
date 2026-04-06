"""Tests for the preprocessing subpackage (data loading, cleaning, ICA)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# data_loading
# ---------------------------------------------------------------------------


def test_load_data_npy(tmp_path):
    """load_data should load a .npy file correctly."""
    from src.preprocessing.data_loading import load_data

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    fpath = tmp_path / "test.npy"
    np.save(fpath, arr)
    loaded = load_data(str(fpath))
    np.testing.assert_array_equal(loaded, arr)


def test_load_data_txt(tmp_path):
    """load_data should load a .txt file correctly."""
    from src.preprocessing.data_loading import load_data

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    fpath = tmp_path / "test.txt"
    np.savetxt(fpath, arr)
    loaded = load_data(str(fpath))
    np.testing.assert_array_almost_equal(loaded, arr)


def test_load_data_pickle(tmp_path):
    """load_data should load a .pickle file correctly."""
    from src.preprocessing.data_loading import load_data

    arr = np.array([1.0, 2.0, 3.0])
    fpath = tmp_path / "test.pickle"
    with open(fpath, "wb") as fh:
        pickle.dump(arr, fh)
    loaded = load_data(str(fpath))
    np.testing.assert_array_equal(loaded, arr)


def test_load_data_unsupported_raises():
    """load_data should raise ValueError for unsupported extensions."""
    from src.preprocessing.data_loading import load_data

    with pytest.raises(ValueError, match="Unsupported"):
        load_data("some_file.csv")


def test_simulate_data_shape():
    """simulate_data should return the correct shape."""
    from src.preprocessing.data_loading import simulate_data

    a = np.eye(5)
    data = simulate_data(a, m=100, iid=False)
    assert data.shape == (5, 100)


def test_simulate_data_iid_shape():
    """simulate_data with iid=True should return the correct shape."""
    from src.preprocessing.data_loading import simulate_data

    a = np.eye(3)
    data = simulate_data(a, m=50, iid=True)
    assert data.shape == (3, 50)


def test_adj_matrix_shape_and_diagonal():
    """adj_matrix should return a square row-normalised matrix with unit diagonal."""
    from src.preprocessing.data_loading import adj_matrix

    a = adj_matrix(15)
    assert a.shape == (15, 15)
    # Diagonal should be non-zero (set to 1 before normalisation)
    for i in range(15):
        assert a[i, i] > 0


# ---------------------------------------------------------------------------
# data_cleaning
# ---------------------------------------------------------------------------


def test_replace_bad_frames_delete():
    """Deleting bad frames should reduce column count by len(bad_frames)."""
    from src.preprocessing.data_cleaning import replace_bad_frames

    arr = np.ones((3, 10))
    bad = np.array([2, 5])
    cleaned = replace_bad_frames(arr, bad, delete_frames=True)
    assert cleaned.shape == (3, 8)


def test_replace_bad_frames_interpolate():
    """Interpolated frames should be the average of adjacent frames."""
    from src.preprocessing.data_cleaning import replace_bad_frames

    arr = np.array([[1.0, 3.0, 5.0, 7.0, 9.0]])
    # Let's pick index 1 → should become (arr[:,0]+arr[:,2])/2 = (1+5)/2 = 3
    arr = np.array([[0.0, 99.0, 4.0, 6.0, 8.0]])
    cleaned = replace_bad_frames(arr, np.array([1]), delete_frames=False)
    expected = (0.0 + 4.0) / 2.0
    assert abs(cleaned[0, 1] - expected) < 1e-10


def test_replace_bad_frames_no_change_to_other():
    """Non-bad frames should be untouched during interpolation."""
    from src.preprocessing.data_cleaning import replace_bad_frames

    arr = np.arange(20, dtype=float).reshape(2, 10)
    original = arr.copy()
    bad = np.array([4])
    cleaned = replace_bad_frames(arr, bad, delete_frames=False)
    for col in range(10):
        if col != 4:
            np.testing.assert_array_equal(cleaned[:, col], original[:, col])


# ---------------------------------------------------------------------------
# ICADecomposition
# ---------------------------------------------------------------------------


def test_ica_decomposition_fit_shape():
    """After fit, ics should have shape [n_vars, n_timepoints]."""
    from src.preprocessing.ica_decomposition import ICADecomposition

    rng = np.random.default_rng(7)
    data = rng.standard_normal((5, 200))
    model = ICADecomposition(
        max_iter=200, tolerance=1e-3, n_comps=5, f_c=1.0, f_s=10.0
    )
    model.fit(data, var_to_keep=0.95, eig_dec=False)
    assert model.ics is not None
    assert model.ics.shape == (5, 200)


def test_ica_decomposition_mixing_mat_shape():
    """Mixing matrix should be square [n_comps × n_comps]."""
    from src.preprocessing.ica_decomposition import ICADecomposition

    rng = np.random.default_rng(8)
    data = rng.standard_normal((4, 150))
    model = ICADecomposition(
        max_iter=200, tolerance=1e-3, n_comps=4, f_c=1.0, f_s=10.0
    )
    model.fit(data, var_to_keep=0.95, eig_dec=False)
    assert model.mixing_mat.shape == (4, 4)


def test_ica_ideal_lp_filter():
    """ideal_lp should return a real array of the requested length."""
    from src.preprocessing.ica_decomposition import ICADecomposition

    model = ICADecomposition(max_iter=200, tolerance=1e-3, n_comps=5, f_c=1.0, f_s=10.0)
    h = model.ideal_lp(m=64)
    assert h.shape == (64,)
    assert np.isrealobj(h)
