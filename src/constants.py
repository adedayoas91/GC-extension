"""Shared constants for the GC-extension package."""

# ---------------------------------------------------------------------------
# Hemisphere colour constants (normalised RGB tuples)
# ---------------------------------------------------------------------------
RGB_LEFT_HEMISPHERE: tuple[float, float, float] = (
    57.0 / 255.0,
    87.0 / 255.0,
    225.0 / 255.0,
)
RGB_RIGHT_HEMISPHERE: tuple[float, float, float] = (
    255.0 / 255.0,
    138.0 / 255.0,
    0.0 / 255.0,
)

# ---------------------------------------------------------------------------
# Default algorithm hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_N_PERM: int = 1000
DEFAULT_N_PASTS: int = 2
DEFAULT_N_LAGS: int = 3
DEFAULT_ALPHA: float = 0.01    # unconditional dependence significance level
DEFAULT_BETA: float = 0.001   # conditional dependence significance level

# ---------------------------------------------------------------------------
# Permutation-test shift bounds
# ---------------------------------------------------------------------------
PERM_SHIFT_MIN: int = 30
PERM_SHIFT_MAX_BUFFER: int = 30

# Tighter bounds used for rising-flank segments (shorter arrays)
RISING_SHIFT_MIN: int = 5
RISING_SHIFT_MAX_BUFFER: int = 8

# ---------------------------------------------------------------------------
# Significance markers
# ---------------------------------------------------------------------------
SIGNIFICANCE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.0001, "****"),
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
)
SIGNIFICANCE_NS: str = "ns"
