"""Tests for the core Granger causality engine (shared utilities)."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------


def test_residual_removes_linear_component():
    """Residual of x ~ x (perfect predictor) should be near zero."""
    from src.core.shared import residual

    rng = np.random.default_rng(0)
    z = rng.standard_normal((1, 100))
    x = z[0] * 2.5 + 1.0          # x is a linear transform of z
    res = residual(x, z)
    assert np.abs(res).mean() < 1e-10


def test_residual_uncorrelated_leaves_signal():
    """Residual when z is independent of x should be close to x itself."""
    from src.core.shared import residual

    rng = np.random.default_rng(1)
    x = rng.standard_normal(100)
    z = rng.standard_normal((1, 100))
    # The regression will capture very little; residual ≈ x
    res = residual(x, z)
    corr = np.corrcoef(x, res)[0, 1]
    assert corr > 0.95


def test_prep_data_shape_zero_lags():
    """prep_data with n_lags=0 should return input unchanged."""
    from src.core.shared import prep_data

    x = np.ones((3, 50))
    assert np.array_equal(prep_data(x, 0), x)


def test_prep_data_shape_nonzero_lags():
    """prep_data with n_lags=2 should stack 3 copies and trim columns."""
    from src.core.shared import prep_data

    n_vars, t = 3, 50
    x = np.random.default_rng(2).standard_normal((n_vars, t))
    out = prep_data(x, 2)
    assert out.shape == (n_vars * 3, t - 2)


def test_ideal_lp_filter_length_and_dtype():
    """ideal_lp_filter should return a real array of the requested length."""
    from src.core.shared import ideal_lp_filter

    h = ideal_lp_filter(f_c=1.0, f_s=10.0, m=64)
    assert h.shape == (64,)
    assert np.isrealobj(h)


def test_combine_fills_single_gaps():
    """combine() should insert the missing index when the gap is exactly 1."""
    from src.core.shared import combine

    result = combine([0, 2, 5, 7])
    assert 1 in result   # gap between 0 and 2 filled
    assert 6 in result   # gap between 5 and 7 filled
    assert 0 in result
    assert 5 in result


def test_combine_no_gaps():
    """combine() with consecutive integers should not insert anything extra."""
    from src.core.shared import combine

    result = combine([1, 2, 3, 4])
    assert result == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# GrangerCausality class
# ---------------------------------------------------------------------------


@pytest.fixture
def gc_model():
    """Return a freshly instantiated GrangerCausality model."""
    from src.core.granger_causality import GrangerCausality

    return GrangerCausality(
        n_perm=10,
        n_pasts=1,
        n_lags=1,
        temporal=True,
        method="cgc",
    )


def test_gc_properties(gc_model):
    """All read-only properties should reflect constructor arguments."""
    assert gc_model.is_time_series is True
    assert gc_model.number_of_perms == 10
    assert gc_model.number_of_pasts == 1
    assert gc_model.number_of_lags == 1


def test_shift_data_no_pasts(gc_model):
    """shift_data with n_pasts=0 should return the array unchanged."""
    gc = gc_model
    gc.n_pasts = 0
    x = np.eye(4)
    assert np.array_equal(gc.shift_data(x), x)


def test_shift_data_shape(gc_model):
    """shift_data should produce the expected stacked-lag shape."""
    gc = gc_model  # n_pasts=1
    n_vars, t = 3, 20
    x = np.random.default_rng(3).standard_normal((n_vars, t))
    result = gc.shift_data(x)
    # expected shape: (n_vars * (n_pasts + 1), t - n_pasts)
    assert result.shape == (n_vars * 2, t - 1)
    assert gc.n_neur == n_vars


def test_number_of_neurons_property(gc_model):
    """number_of_neurons should equal n_vars after shift_data."""
    gc = gc_model
    n_vars, t = 4, 30
    x = np.random.default_rng(4).standard_normal((n_vars, t))
    gc.shifted_data = gc.shift_data(x)
    assert gc.number_of_neurons == n_vars


def test_compute_confusion_matrix_perfect(gc_model):
    """All-ones inferred and ground truth should yield TP=n², FP=FN=TN=0."""
    gc = gc_model
    n = 4
    gc.conn_mat = np.ones((n, n))
    cm = gc.compute_confusion_matrix(np.ones((n, n)), simulation=False)
    tp, fp, fn, tn = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    assert tp == n * n
    assert fp == fn == tn == 0


def test_compute_confusion_matrix_simulation_transposes(gc_model):
    """simulation=True should transpose the ground truth before comparison."""
    gc = gc_model
    # asymmetric ground truth: only (0,1) is present
    a = np.zeros((3, 3))
    a[0, 1] = 1
    gc.conn_mat = np.zeros((3, 3))
    gc.conn_mat[1, 0] = 1   # matches a.T[0,1]
    cm = gc.compute_confusion_matrix(a, simulation=True)
    # After transpose, GT has a[1,0]=1; conn_mat[1,0]=1 → TP=1
    assert int(cm[0, 0]) == 1


def test_all_metrics_values():
    """Verify metric formulas with a hand-crafted confusion matrix."""
    from src.core.granger_causality import GrangerCausality

    gc = GrangerCausality(n_perm=10, n_pasts=1, n_lags=1, temporal=True, method="cgc")
    # TP=3, FP=1, FN=1, TN=5
    gc.confusion_matrix = np.array([[3, 1], [1, 5]])
    metrics = gc.all_metrics()
    # accuracy = (3+5)/10 = 0.8
    assert abs(metrics[0] - 0.8) < 1e-9
    # precision = 3/(3+1) = 0.75
    assert abs(metrics[1] - 0.75) < 1e-9
    # recall = 3/(3+1) = 0.75
    assert abs(metrics[2] - 0.75) < 1e-9


def test_get_connectivity_matrix_shape(gc_model):
    """get_connectivity_matrix should return a square [n_vars × n_vars] matrix."""
    gc = gc_model
    rng = np.random.default_rng(5)
    n_vars, t = 3, 40
    x = rng.standard_normal((n_vars, t))
    gc.fit(x, verbose=0)
    conn = gc.get_connectivity_matrix(simulation=False)
    assert conn.shape == (n_vars, n_vars)


# ---------------------------------------------------------------------------
# compare_with_gt
# ---------------------------------------------------------------------------


def test_compare_with_gt_values():
    """Check the four colour codes for TP / FP / FN / TN."""
    from src.core.granger_causality import compare_with_gt

    a = np.array([[1, 0], [0, 0]])
    inf = np.array([[1, 1], [0, 0]])
    result = compare_with_gt(a, inf, simulation=False)
    assert result[0, 0] == 40   # TP
    assert result[0, 1] == 30   # FP
    assert result[1, 0] == 20   # FN (a!=0 but inf==0 → no, a==0 → FN impossible)
    assert result[1, 1] == 10   # TN
