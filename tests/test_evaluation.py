"""Tests for the evaluation subpackage (metrics and visualisation)."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ComputeMetrics
# ---------------------------------------------------------------------------


def test_compute_confusion_matrix_all_correct():
    """Perfect prediction: all non-zero → all TP, TN when gt matches."""
    from src.evaluation.metrics import ComputeMetrics

    n = 4
    mat = np.ones((n, n))
    m = ComputeMetrics(conn_mat=mat, ground_truth=mat.copy(), n_pasts=1)
    cm = m.compute_confusion_matrix()
    assert int(cm[0, 0]) == n * n   # TP
    assert int(cm[1, 1]) == 0       # TN


def test_compute_confusion_matrix_all_wrong():
    """Worst case: inferred all ones, GT all zeros → all FP."""
    from src.evaluation.metrics import ComputeMetrics

    n = 3
    inf = np.ones((n, n))
    gt = np.zeros((n, n))
    m = ComputeMetrics(conn_mat=inf, ground_truth=gt, n_pasts=1)
    cm = m.compute_confusion_matrix()
    assert int(cm[1, 0]) == n * n   # FP
    assert int(cm[0, 0]) == 0       # TP


def test_metrics_from_confusion_matrix():
    """Validate metric formulas with a hand-crafted confusion matrix."""
    from src.evaluation.metrics import ComputeMetrics

    # TP=6, FN=2, FP=1, TN=11  (layout: [[TP,FN],[FP,TN]])
    cm = np.array([[6, 2], [1, 11]])
    m = ComputeMetrics(
        conn_mat=np.zeros((2, 2)),
        ground_truth=np.zeros((2, 2)),
        n_pasts=1,
    )
    m.confusion_matrix = cm
    metrics = m.compute_metrics_from_confusion_matrix()
    # accuracy = (6+11)/(6+2+1+11) = 17/20 = 0.85
    assert abs(metrics[0] - 0.85) < 1e-9
    # precision = 6/(6+1) ≈ 0.857
    assert abs(metrics[1] - 6 / 7) < 1e-9
    # recall = 6/(6+2) = 0.75
    assert abs(metrics[2] - 0.75) < 1e-9
    # FPR = 1/(1+11) ≈ 0.0833
    assert abs(metrics[3] - 1 / 12) < 1e-9


def test_repopulate_shape():
    """repopulate should produce a square full-population matrix."""
    from src.evaluation.metrics import ComputeMetrics

    n_full = 10
    n_sub = 4
    inf = np.eye(n_sub)
    traces = np.zeros((n_full, 100))
    idx = np.array([0, 2, 4, 6])
    result = ComputeMetrics.repopulate(inf, traces, idx)
    assert result.shape == (n_full, n_full)
    # The diagonal entries of the sub-population should be mapped
    for i in range(n_sub):
        assert result[idx[i], idx[i]] == 1


def test_roi_neighbors_returns_dicts():
    """roi_neighbors should return two dicts keyed by ROI index."""
    from src.evaluation.metrics import ComputeMetrics

    conn = np.array([[0, 1, 0],
                     [0, 0, 1],
                     [0, 0, 0]], dtype=float)
    m = ComputeMetrics(conn_mat=conn, ground_truth=conn, n_pasts=1)
    out_n, in_n = m.roi_neighbors()
    assert set(out_n.keys()) == {0, 1, 2}
    assert 1 in out_n[0]   # 0→1 exists
    assert 2 in out_n[1]   # 1→2 exists
    assert 0 in in_n[1]    # 0→1 means 1 has in-neighbor 0


# ---------------------------------------------------------------------------
# pval_to_star
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pval, expected",
    [
        (0.00001, "****"),
        (0.0005, "***"),
        (0.005, "**"),
        (0.03, "*"),
        (0.1, "ns"),
    ],
)
def test_pval_to_star(pval, expected):
    from src.evaluation.visualization import pval_to_star

    assert pval_to_star(pval) == expected


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def test_get_coords_shape():
    """get_coords should return [tot, 2] array."""
    from src.evaluation.visualization import get_coords

    coords = get_coords(mid=3, tot=6)
    assert coords.shape == (6, 2)


def test_get_label_and_color_lists_lengths():
    """Label and colour lists should both have length tot."""
    from src.evaluation.visualization import get_label_and_color_lists

    labels, colors = get_label_and_color_lists(mid=4, tot=8)
    assert len(labels) == 8
    assert len(colors) == 8
