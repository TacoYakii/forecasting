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

- **Python interpreter:** `.venv/bin/python`
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

The concrete ForecastResult type and shape depend on the model family and forecasting method:

**Statistical Models** (RollingRunner — StatefulPredictor, recursive forecast → update_state loop):

| Model | Method | Single Call | Final (after Runner) |
|-------|--------|-------------|----------------------|
| ARIMA-GARCH | `forecast(horizon)` | `ParametricForecastResult` (1, H) | **(N, H)** |
| SARIMA-GARCH | `forecast(horizon)` | `ParametricForecastResult` (1, H) | **(N, H)** |
| ARFIMA-GARCH | `forecast(horizon)` | `ParametricForecastResult` (1, H) | **(N, H)** |
| GarchBase (all) | `simulate_paths(n_paths, horizon)` | `SampleForecastResult` (1, n_paths, H) | **(N, n_paths, H)** |

**Machine Learning Models** (PerHorizonRunner — independent model per horizon):

| Model | Method | Single Call | Final (after Runner) |
|-------|--------|-------------|----------------------|
| XGBoost, CatBoost, NGBoost, PGBM, LR | `forecast(X, idx)` | `ParametricForecastResult` (T, 1) | **(N_common, H)** |

- T = test observations per horizon CSV, N_common = intersection of test indices across all H horizons

**Deep Time Series Models** (RollingRunner — ContextPredictor, rolling context window):

| Model | Loss | Single Call | Final |
|-------|------|-------------|-------|
| DeepAR / TFT | `DistributionLoss` | `ParametricForecastResult` (1, H) | **(N, H)** |
| DeepAR / TFT | `MQLoss` / `IQLoss` | `QuantileForecastResult` (1, H) | **(N, H)** |

**Foundation Models** (RollingRunner — ContextPredictor, Monte Carlo sampling):

| Model | Single Call | Final |
|-------|-------------|-------|
| Chronos | `SampleForecastResult` (1, n_samples, H) | **(N, n_samples, H)** |
| Moirai | `SampleForecastResult` (1, n_samples, H) | **(N, n_samples, H)** |

Two Runner patterns orchestrate aggregation:
- **`RollingRunner`** — Each step returns a single origin result, concatenated along axis=0 to produce (N, ...)
- **`PerHorizonRunner`** — Each horizon model returns (T, 1), stacked along axis=1 to produce (N_common, H)

### Model Families (`src/models/`)

| Family | Models | Runner | Output |
|--------|--------|--------|--------|
| `statistical/` | ARIMA-GARCH, SARIMA-GARCH, ARFIMA-GARCH | RollingRunner (Stateful) | ParametricForecastResult / SampleForecastResult |
| `machine_learning/` | CatBoost, NGBoost, PGBM, XGBoost, LR | PerHorizonRunner | ParametricForecastResult |
| `deep_time_series/` | DeepAR, TFT (NeuralForecast) | RollingRunner (Context) | DistributionLoss → Parametric, MQLoss → Quantile |
| `foundation/` | Chronos, Moirai | RollingRunner (Context) | SampleForecastResult |

### Evaluation (`src/utils/`)

- **nMAPE**: KPX day-ahead and real-time market evaluation

### Exogenous Variable Classification

Exogenous variables are classified by **availability at prediction time**:

| Category | Definition | Example | Used by |
|----------|-----------|---------|---------|
| **`futr_exog`** | Variables with known future values | NWP forecasts (wind speed, temperature) | Statistical, Deep |
| **`hist_exog`** | Only past observations available | Measured wind speed (SCADA) | Deep only |

Exogenous interface by model family:

| Family | Parameter | Reason |
|--------|-----------|--------|
| Statistical (GARCH) | `exog_cols` | Only futr used for training/prediction (hist causes leakage) |
| ML (CatBoost, etc.) | `exog_cols` | Feature engineering completed in per-horizon CSV |
| Deep (DeepAR, TFT) | `futr_cols` + `hist_cols` | NeuralForecast requires separate `futr_exog_list`/`hist_exog_list` |
| Foundation (Moirai) | `exog_cols` | Used as `past_feat_dynamic_real` (context only) |
| Foundation (Chronos) | — | No exogenous support |

Runner passes only **futr_exog** to the future horizon window (prevents hist leakage).
Deep model encoders leverage both futr+hist in the past window, compressing into hidden state.

> Design details: `docs/exogenous_variable_design.md`

## Key Patterns

- **ForecastResult -> Distribution**: All Result objects can be converted to a Distribution object for a specific horizon via `to_distribution(h)`. Marginal statistics (`mean`, `std`, `ppf`, `interval`) are only available on Distribution — ForecastResult delegates via `to_distribution(h)`. Design details: `docs/forecast_result_distribution_design.md`
- **Distribution Registry**: Extensible mapping of distribution name -> scipy distribution + parameter names. Used only by MLE-based models; moment matching uses `mu_std_to_dist_params()`
- **Experiment directory**: Auto-incrementing under `res/{ModelClass}/{exp_num}/`, stores logs, metadata, and serialized models

## Design Principles

- **Single Responsibility**: Each class has one responsibility. e.g., `ForecastResult` handles result storage/conversion only; `Distribution` handles probabilistic operations (`mean`, `ppf`, etc.). Do not duplicate distribution operations in Result — delegate via `to_distribution(h)`.
- **Open-Closed**: Extend via Registry registration or subclassing, not by modifying existing code.
- **Liskov Substitution**: `ForecastResult` subclasses (`Parametric`, `Quantile`, `Sample`) must be interchangeable through the parent interface.
- **Interface Segregation**: Models implement only the interface they need (e.g., `StatefulPredictor` vs `ContextPredictor`).
- **Dependency Inversion**: Runners depend on Predictor abstractions, not concrete model classes.

## Conventions

- Code and docstrings in English; Korean terms used in domain-specific contexts (KPX market types, nMAPE)
- Every model's `forecast()` returns a ForecastResult object (`ParametricForecastResult`, `QuantileForecastResult`, `SampleForecastResult`) — never raw mu/std
- `etc/` contains legacy pre-refactoring code and scripts (no longer in use); `res/` stores experiment results (both gitignored)
- Notebooks in `notebooks/` are for exploration and testing, not production code
- When adding tests, update the corresponding `TEST_*.md` file in the test directory to document what each test verifies (e.g., `docs/TEST_EXOG_SPLIT.md`)
