# GC-extension

A Python package that re-examines Granger causality through the lens of **Causal Bayesian Networks (CBNs)** and **Reichenbach's Common Cause Principle (RCCP)**. The library provides two complementary algorithms — conventional/full-conditioning Granger causality (`cGC` / `fcGC`) and a rising-flank variant designed for calcium-imaging data — and ships with evaluation utilities and preprocessing helpers.

---

## Table of contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core API](#core-api)
- [Example notebooks](#example-notebooks)
- [Development](#development)
- [Tests](#tests)

---

## Overview

Classical Granger causality detects predictive, lagged dependencies between time series, but does not distinguish direct causal links from shared-cause confounds. This package augments the standard test with:

1. **Conditional independence testing** — each candidate edge is tested both unconditionally (marginal correlation) and conditionally (partial correlation after regressing out the remaining variables), following CBN d-separation rules.
2. **Permutation-based p-values** — circular-shift null distributions computed via Numba-JIT routines for speed.
3. **Rising-flank restriction** — an optional mode that restricts the analysis to the rising flanks of calcium-transient waveforms to improve signal-to-noise in two-photon imaging data.

Two conditioning strategies are supported:

| Method | Flag | Description |
|--------|------|-------------|
| Conditional GC | `"cgc"` | Conditions on the own past of each variable |
| Full-conditioning GC | `"fcgc"` | Conditions on the full remaining variable set |

---

## Repository layout

```
GC-extension/
├── src/
│   ├── core/
│   │   ├── granger_causality.py   # causalisedGrangerCausality — main class
│   │   ├── rising_flanks_gc.py    # RisingFlankGrangerCausality
│   │   └── shared.py              # Numba-JIT numerical utilities
│   ├── preprocessing/
│   │   ├── data_cleaning.py       # Bad-frame removal / interpolation
│   │   ├── data_loading.py        # .mat / NumPy data loaders
│   │   └── ica_decomposition.py   # ICA helpers
│   ├── evaluation/
│   │   ├── metrics.py             # ComputeMetrics (confusion matrix, F1, …)
│   │   └── visualization.py       # Topography & connectivity-matrix plots
│   └── constants.py               # Package-wide hyper-parameter defaults
├── examples/                      # Jupyter notebooks (see below)
├── tests/                         # pytest test suite
├── pyproject.toml                 # Build metadata & dependency pins
├── uv.lock                        # Reproducible dependency lock file (uv)
├── environment.yml                # Conda environment (alternative)
└── Makefile                       # Developer shortcuts
```

---

## Installation

### Recommended — uv

[uv](https://github.com/astral-sh/uv) resolves dependencies from the included `uv.lock` file, giving a fully reproducible environment.

```bash
# clone
git clone https://github.com/adedayoas91/GC-extension.git
cd GC-extension

# create a virtual environment and install (editable + dev extras)
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
# or simply
make install
```

The optional `cdt` extra (for Structural Hamming Distance via the `causal-learn` / CDT toolkit) can be installed separately:

```bash
uv pip install -e ".[cdt]"
```

### Alternative — conda

```bash
conda env create --name gc-extension -f environment.yml
conda activate gc-extension
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
from src.core.granger_causality import causalisedGrangerCausality

# Simulate 10 variables × 500 time points
rng = np.random.default_rng(0)
X = rng.standard_normal((10, 500))

# Fit the conditional GC model
model = causalisedGrangerCausality(
    n_perm=200,     # permutations per pair
    n_pasts=2,      # time lags to include
    n_lags=2,       # maximum connectivity lag
    temporal=True,  # data are a time series
    method="cgc",   # "cgc" or "fcgc"
)
model.fit(X, verbose=0)

# Retrieve the weighted connectivity matrix
conn = model.get_connectivity_matrix(simulation=False, alpha=0.01, beta=0.001)
print(conn.shape)  # (10, 10)
```

### Rising-flank variant (calcium imaging)

```python
from src.core.rising_flanks_gc import RisingFlankGrangerCausality, cut_rising_flanks_lp

# Extract rising flanks per neuron
idx = [cut_rising_flanks_lp(X[i], f_c=1.0, f_s=30.0, m=256)[1] for i in range(X.shape[0])]

rf_model = RisingFlankGrangerCausality(
    n_perm=100, n_pasts=1, n_lags=1, f_s=30.0, seg_len=20
)
rf_model.fit_rising(X, idx, verbose=0)
```

---

## Core API

### `causalisedGrangerCausality`

| Method | Description |
|--------|-------------|
| `fit(X, verbose)` | Fit unconditional and conditional correlation matrices |
| `get_connectivity_matrix(simulation, alpha, beta)` | Threshold results into a weighted adjacency matrix |
| `compute_confusion_matrix(A, simulation)` | Compare inferred matrix against ground truth |
| `all_metrics()` | Accuracy, precision, recall, FPR, balanced accuracy, F1 |
| `compute_shd_sid(A, inf, simulation)` | Structural Hamming Distance (requires `cdt`) |
| `fit_rising(X, idx)` | Rising-flank fit (delegates to internal helpers) |
| `plot_extended_connectivity_matrix(alpha, beta)` | Visualise per-lag connectivity blocks |

### `RisingFlankGrangerCausality`

| Method | Description |
|--------|-------------|
| `fit_rising(X, idx, verbose)` | Fit GC restricted to rising-flank segments |
| `get_conditioning_set(i, j)` | Build the conditioning set for a variable pair |
| `ideal_lp(f_c, m)` | Ideal low-pass FIR filter kernel |

### Shared utilities (`src.core.shared`)

| Function | Description |
|----------|-------------|
| `perm_test(x, y, n_perm)` | Numba-JIT permutation p-value |
| `residual(x, z)` | Linear regression residuals (conditioning) |
| `cross_corr(x, y, n_lags)` | Absolute lagged cross-correlation |
| `prep_data(x, n_lags)` | Build a time-lagged design matrix |
| `ideal_lp_filter(f_c, f_s, m)` | Ideal low-pass FIR kernel |

---

## Example notebooks

All notebooks live in `examples/` and cover several benchmark datasets:

| Notebook | Dataset / scenario |
|----------|--------------------|
| `simulation_SingleLag_cGC.ipynb` | Simulated single-lag data — cGC |
| `simulation_SingleLag_fcGC.ipynb` | Simulated single-lag data — fcGC |
| `simulation_VarLags_cGC.ipynb` | Simulated variable-lag data — cGC |
| `simulation_VarLags_fcGC-.ipynb` | Simulated variable-lag data — fcGC |
| `Lorenz-data-p10.ipynb` | Lorenz system (p = 10) |
| `Lorenz-data-p40.ipynb` | Lorenz system (p = 40) |
| `Sachs_data-cGC.ipynb` | Sachs protein-signalling benchmark — cGC |
| `Sachs_data-fcgc.ipynb` | Sachs protein-signalling benchmark — fcGC |
| `NTS-NOTEARS_data.ipynb` | NTS / NOTEARS benchmark |
| `TCDF_data.ipynb` | TCDF benchmark |

---

## Development

```bash
# lint
make lint        # ruff
make pylint      # pylint (threshold: 7.0)

# test + coverage
make test
make coverage
```

The `Makefile` provides `make all` as a convenience target that runs `install → lint → test` in sequence.

---

## Tests

```bash
pytest            # uses settings from pyproject.toml [tool.pytest.ini_options]
pytest --cov=src  # with coverage report
```

The test suite lives in `tests/` and covers core algorithm logic, preprocessing helpers, evaluation metrics, and rising-flank routines.
