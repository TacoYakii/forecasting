# Project: Windpower Forecasting 

Python 3.13 based probabilistic wind power forecasting system. 

## Git
- Do NOT append `Co-Authored-By` lines to commit messages

## Code Style

### Linting (ruff)
- **E** (pycodestyle errors) + **F** (pyflakes): standard PEP 8 and logical error checks
- **I** (isort): import ordering
- **D** (pydocstyle): docstring rules, with `D100` (module-level) and `D104` (`__init__.py`) ignored
- Line length: 88
- Per-file ignores: `tests/` skips D rules; `notebooks/` skips D and E rules

### Docstrings
- Use **Google Style** docstrings for all classes and functions
- Class docstrings should include sections relevant to the domain
  (e.g., `Loss selection:`, `Model-specific hyperparameters:`)
- Always include an `Example:` section with `>>>` usage

### Build
- Build backend: **hatchling** (not setuptools)
- Package manager: **uv**


## Development Setup

- **Python interpreter:** `.venv/bin/python` (absolute: `.venv/bin/python`)
- Run commands via `.venv/bin/python -m <module>` (e.g., `.venv/bin/python -m pytest`)

```bash
uv sync          # install dependencies
uv run python    # run in project environment
uv run pytest    # run tests
```

- Python 3.13 (managed by `uv`)
- Key dependencies:
  - **Core:** NumPy, Pandas, PyTorch, SciPy
  - **ML models:** scikit-learn, CatBoost, NGBoost, XGBoost, PGBM
  - **Deep time-series:** NeuralForecast
  - **Foundation models:** Chronos, Uni2TS (Moirai), GluonTS
  - **Meteorological / geospatial:** MetPy, xarray, ecmwfapi, Folium
  - **Utilities:** tqdm, joblib, Optuna, requests, python-dotenv, Matplotlib, PyYAML
  - **High-performance compute:** Numba, JAX

## Architecture
- '/docs': Documentation of the project
- '/src': Source code of the project
- '/data': Data files of the project
- '/etc': Configuration files of the project
- '/res': Result files of the project
- '/notebooks': Notebook files of the project

### Core Abstraction (`src/core/`)
Every model's `forecast()` returns a **ForecastResult** object. Each Result can be converted to a **Distribution** object via `to_distribution(h)`.

The concrete ForecastResult type is determined by the model's **training objective**:

```
MLE-based:      Model.forecast() → ParametricForecastResult (1, H) or (N, 1)  → Runner concat → (N, H)
Quantile-based: Model.forecast() → QuantileForecastResult   (1, H)            → Runner concat → (N, H)
Sample-based:   Model.forecast() → SampleForecastResult     (1, n_samples, H) → Runner concat → (N, n_samples, H)
```

- **MLE-based — `ParametricForecastResult`:** The model assumes a parametric distribution and estimates its native parameters (e.g., mu/sigma for Normal). The **Distribution Registry** (Normal, StudentT, Gamma, Weibull, ...) maps distribution names to scipy distributions + parameter names.
  - Statistical models (ARIMA-GARCH, etc.), ML models (NGBoost, CatBoost, PGBM), deep models with DistributionLoss
- **Quantile-based — `QuantileForecastResult`:** The model directly predicts values at specific quantile levels, without assuming a distributional form.
  - Deep models with MQLoss / IQLoss (DeepAR, TFT)
- **Sample-based — `SampleForecastResult`:** The model generates draws (sample paths) from the predictive distribution, representing it empirically. This covers both Monte Carlo sampling from generative models and forward simulation of stochastic processes.
  - Foundation models (Chronos, Moirai) via Monte Carlo sampling, GARCH family via simulated sample paths

Two Runner patterns orchestrate aggregation depending on the forecasting scheme:
- **`RollingRunner`** — Recursive forecasting: forecast -> update_state loop. Each call returns (1, H) -> stacked along axis=0 to produce (N, H)
- **`PerHorizonRunner`** — Cross-sectional forecasting: independent models per horizon predict all time points at once. Each call returns (N, 1) -> stacked along axis=1 to produce (N, H)

### Model Families (`src/models/`)

| Family | Models | Runner | Output |
|--------|--------|--------|--------|
| `statistical/` | ARIMA-GARCH, SARIMA, ARFIMA | RollingRunner | ParametricForecastResult |
| `machine_learning/` | CatBoost, NGBoost, PGBM, XGBoost, LR | PerHorizonRunner | ParametricForecastResult |
| `deep_time_series/` | DeepAR, TFT (NeuralForecast) | NeuralForecast wrapper | DistributionLoss -> ParametricForecastResult, MQLoss -> QuantileForecastResult |
| `foundation/` | Chronos, Moirai | from_pretrained() | SampleForecastResult |

### Data Pipeline (`src/data/`)

Download (KMA, ECMWF) -> NWP preprocessing (GRIB2/TXT readers, validators, derived variables) -> Training data builder (continuous/per-horizon, lag/exogenous features)

### Hierarchical Forecasting (`src/pipelines/`)

`BaseForecastRunner` -> `HierarchyForecastCoordinator` (selects best model per level/horizon by CRPS) -> `HierarchyForecastSampler` (generates ensemble datasets). The S matrix defines the aggregation structure.

### Evaluation (`src/utils/`)

- **nMAPE**: KPX day-ahead and real-time market evaluation
- **CRPS**: Probabilistic calibration metrics
- **Loss Registry**: Pluggable loss functions (RandomCRPSLoss, QuantileCRPSLoss, PinballLoss)

## Key Patterns

- **ForecastResult -> Distribution**: All Result objects can be converted to a Distribution object for a specific horizon via `to_distribution(h)` (ParametricDistribution or EmpiricalDistribution)
- **Distribution Registry**: Extensible mapping of distribution name -> scipy distribution + parameter names. Used only by MLE-based models; moment matching uses `mu_std_to_dist_params()`
- **Experiment directory**: Auto-incrementing under `res/{ModelClass}/{exp_num}/`, stores logs, metadata, and serialized models

## Conventions

- Code and docstrings in English; Korean terms used in domain-specific contexts (KPX market types, nMAPE)
- Every model's `forecast()` returns a ForecastResult object (`ParametricForecastResult`, `QuantileForecastResult`, `SampleForecastResult`) — never raw mu/std
- `etc/` contains config files and auxiliary scripts; `res/` stores experiment results (both gitignored)
- Notebooks in `notebooks/` are for exploration and testing, not production code
