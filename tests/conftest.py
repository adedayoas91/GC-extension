"""Shared pytest fixtures for the GC-extension test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_timeseries(rng) -> np.ndarray:
    """Small synthetic time series: 4 variables × 200 time-steps."""
    return rng.standard_normal((4, 200))


@pytest.fixture(scope="session")
def chain_adjacency() -> np.ndarray:
    """5-node chain adjacency matrix: 1→2→3→4→5."""
    a = np.zeros((5, 5))
    for i in range(4):
        a[i + 1, i] = 1.0
    return a
