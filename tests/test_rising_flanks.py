"""Tests for the rising-flank GC module."""

from __future__ import annotations

import numpy as np


def test_rising_flank_gc_shift_data_shape():
    """shift_data should produce expected shape for n_pasts=2."""
    from src.core.rising_flanks_gc import RisingFlankGrangerCausality

    model = RisingFlankGrangerCausality(
        n_perm=10, n_pasts=2, n_lags=1, f_s=10.0, seg_len=50
    )
    x = np.random.default_rng(10).standard_normal((3, 30))
    result = model.shift_data(x)
    # n_vars * (n_pasts + 1) rows, t - n_pasts columns
    assert result.shape == (3 * 3, 30 - 2)


def test_rising_flank_gc_moving_average_length():
    """moving_average should return len(data) - win_size + 1 values."""
    from src.core.rising_flanks_gc import RisingFlankGrangerCausality

    model = RisingFlankGrangerCausality(
        n_perm=10, n_pasts=1, n_lags=1, f_s=10.0, seg_len=50
    )
    data = np.arange(10, dtype=float)
    avg = model.moving_average(data, win_size=3)
    assert len(avg) == 8


def test_rising_flank_gc_ideal_lp_length():
    """ideal_lp should return an array of the requested length."""
    from src.core.rising_flanks_gc import RisingFlankGrangerCausality

    model = RisingFlankGrangerCausality(
        n_perm=10, n_pasts=1, n_lags=1, f_s=10.0, seg_len=50
    )
    h = model.ideal_lp(f_c=1.0, m=64)
    assert h.shape == (64,)
    assert np.isrealobj(h)


def test_cut_rising_flanks_lp_returns_tuple():
    """cut_rising_flanks_lp should return (values, indices) tuple."""
    from src.core.rising_flanks_gc import cut_rising_flanks_lp

    rng = np.random.default_rng(11)
    arr = rng.standard_normal(200)
    vals, idx = cut_rising_flanks_lp(arr, f_c=1.0, f_s=10.0, m=64)
    assert len(vals) == len(idx)


def test_group_consecutive_indices():
    """group_consecutive_indices should separate non-adjacent groups."""
    from src.core.rising_flanks_gc import group_consecutive_indices

    row = np.array([0, 1, 2, 5, 6, 10])
    groups = group_consecutive_indices(row)
    assert len(groups) == 3
    assert groups[0] == [0, 1, 2]
    assert groups[1] == [5, 6]
    assert groups[2] == [10]


def test_matrix_difference_same_returns_false():
    """matrix_difference should set all matching positions to False."""
    from src.core.rising_flanks_gc import matrix_difference

    a = np.ones((3, 3))
    b = np.ones((3, 3))
    result = matrix_difference(a, b)
    # Every entry matches → all False
    assert not np.any(result)


def test_matrix_difference_all_different():
    """When no entries match, the output equals the original."""
    from src.core.rising_flanks_gc import matrix_difference

    a = np.ones((3, 3))
    b = np.zeros((3, 3))
    result = matrix_difference(a, b)
    np.testing.assert_array_equal(result, a)
