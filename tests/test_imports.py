"""Tests for backward-compatibility aliases in src/__init__.py."""

from __future__ import annotations


def test_primary_class_name():
    """The primary class should be importable as causalisedGrangerCausality."""
    from src.core.granger_causality import causalisedGrangerCausality

    assert causalisedGrangerCausality.__name__ == "causalisedGrangerCausality"


def test_backward_compat_gcstar_alias():
    """GcStar should be an alias for causalisedGrangerCausality."""
    import src
    from src.core.granger_causality import causalisedGrangerCausality

    assert src.GcStar is causalisedGrangerCausality


def test_backward_compat_granger_causality_alias():
    """GrangerCausality (pre-rename alias) should resolve to the same class."""
    import src
    from src.core.granger_causality import causalisedGrangerCausality

    assert src.GrangerCausality is causalisedGrangerCausality


def test_backward_compat_rising_flanks_alias():
    """RisingFlanks should be an alias for RisingFlankGrangerCausality."""
    import src
    from src.core.rising_flanks_gc import RisingFlankGrangerCausality

    assert src.RisingFlanks is RisingFlankGrangerCausality


def test_backward_compat_ica_dec_alias():
    """ICA_dec should be an alias for ICADecomposition."""
    import src
    from src.preprocessing.ica_decomposition import ICADecomposition

    assert src.ICA_dec is ICADecomposition


def test_backward_compat_replace_nan_alias():
    """replace_nan should be an alias for replace_bad_frames."""
    import src
    from src.preprocessing.data_cleaning import replace_bad_frames

    assert src.replace_nan is replace_bad_frames


def test_backward_compat_compare_with_GT_alias():
    """compare_with_GT should be an alias for compare_with_gt."""
    import src
    from src.core.granger_causality import compare_with_gt

    assert src.compare_with_GT is compare_with_gt


def test_top_level_imports_work():
    """All documented public names should be importable from ``src``."""
    import src

    public_names = [
        "causalisedGrangerCausality",
        "RisingFlankGrangerCausality",
        "ICADecomposition",
        "ComputeMetrics",
        "TopographyVisualizer",
        "load_data",
        "simulate_data",
        "adj_matrix",
        "replace_bad_frames",
        "perm_test",
        "perm_test_shift",
        "cross_corr",
        "residual",
        "prep_data",
        "ideal_lp_filter",
        "combine",
        "pval_to_star",
    ]
    for name in public_names:
        assert hasattr(src, name), f"src.{name} not found"
