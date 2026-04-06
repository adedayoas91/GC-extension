"""GC-extension: Granger causality with Causal Bayesian networks.

Top-level package that re-exports the primary API and provides
backward-compatibility aliases for legacy code that used the old
flat-module imports from ``src/d_CSL.py``, ``src/rising_flanks.py``, etc.

Usage
-----
>>> from src import causalisedGrangerCausality, RisingFlankGrangerCausality
"""

from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_N_LAGS,
    DEFAULT_N_PASTS,
    DEFAULT_N_PERM,
)
from .core import (
    RisingFlankGrangerCausality,
    causalisedGrangerCausality,
    combine,
    compare_with_gt,
    cross_corr,
    cut_rising_flanks_lp,
    group_consecutive_indices,
    ideal_lp_filter,
    matrix_difference,
    perm_test,
    perm_test_shift,
    prep_data,
    residual,
)
from .evaluation import (
    ComputeMetrics,
    TopographyVisualizer,
    get_coords,
    get_label_and_color_lists,
    plot_background_and_cells,
    plot_directed_graph,
    plot_drive,
    plot_gc_matrix,
    pval_to_star,
)
from .preprocessing import (
    ICADecomposition,
    adj_matrix,
    load_data,
    replace_bad_frames,
    simulate_data,
)

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (old snake_case / CamelCase names)
# ---------------------------------------------------------------------------

# Classes
GcStar = causalisedGrangerCausality          # was d_CSL.GcStar
GrangerCausality = causalisedGrangerCausality  # pre-rename alias
RisingFlanks = RisingFlankGrangerCausality   # was rising_flanks.RisingFlanks
ICA_dec = ICADecomposition                   # was ICA_dec.ICA_dec
Compute_metrics = ComputeMetrics             # was compute_metrics.Compute_metrics
Visualize_on_topography = TopographyVisualizer  # was compute_metrics.Visualize_on_topography

# Functions
compare_with_GT = compare_with_gt            # was d_CSL.compare_with_GT
replace_nan = replace_bad_frames             # was dataPreProcessing.replace_nan
adj_mtx = adj_matrix                         # was dataload.adj_mtx

__all__ = [
    # Primary class (new name)
    "causalisedGrangerCausality",
    # Other new names
    "RisingFlankGrangerCausality",
    "ICADecomposition",
    "ComputeMetrics",
    "TopographyVisualizer",
    "compare_with_gt",
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
    "cut_rising_flanks_lp",
    "group_consecutive_indices",
    "matrix_difference",
    "pval_to_star",
    "get_label_and_color_lists",
    "get_coords",
    "plot_gc_matrix",
    "plot_directed_graph",
    "plot_background_and_cells",
    "plot_drive",
    # Constants
    "DEFAULT_N_PERM",
    "DEFAULT_N_PASTS",
    "DEFAULT_N_LAGS",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    # Backward-compat aliases
    "GcStar",
    "GrangerCausality",
    "RisingFlanks",
    "ICA_dec",
    "Compute_metrics",
    "Visualize_on_topography",
    "compare_with_GT",
    "replace_nan",
    "adj_mtx",
]
