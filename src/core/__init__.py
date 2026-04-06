"""Core subpackage: Granger causality engines."""

from .granger_causality import GrangerCausality, compare_with_gt
from .rising_flanks_gc import (
    RisingFlankGrangerCausality,
    cut_rising_flanks_lp,
    group_consecutive_indices,
    matrix_difference,
)
from .shared import (
    combine,
    cross_corr,
    ideal_lp_filter,
    perm_test,
    perm_test_shift,
    prep_data,
    residual,
)

__all__ = [
    # Main classes
    "GrangerCausality",
    "RisingFlankGrangerCausality",
    # Compare / visualise helper
    "compare_with_gt",
    # Rising-flank utilities
    "cut_rising_flanks_lp",
    "group_consecutive_indices",
    "matrix_difference",
    # Shared numerical helpers
    "perm_test",
    "perm_test_shift",
    "cross_corr",
    "residual",
    "prep_data",
    "ideal_lp_filter",
    "combine",
]
