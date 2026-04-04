# Timeseries Forecasting

A Python 3.13 probabilistic wind power forecasting system.

Provides a unified interface across multiple model families (statistical, machine learning, deep learning, foundation models). Every model returns a probabilistic forecast via the `ForecastResult` abstraction. Multiple forecasts can be combined via forecast combining, and non-parametric density estimation is supported through Conditional Kernel Density (CKD).

## Architecture

```
src/
├── core/                  # Core abstractions (BaseModel, ForecastResult, Distribution, Runner, CKDRunner)
├── models/
│   ├── statistical/       # ARIMA-GARCH, SARIMA-GARCH, ARFIMA-GARCH
│   ├── machine_learning/  # XGBoost, CatBoost, NGBoost, PGBM, LR
│   ├── deep_time_series/  # DeepAR, TFT (NeuralForecast)
│   ├── foundation/        # Chronos, Moirai (pretrained, with fine-tuning support)
│   ├── combining/         # Forecast combining (Vertical, Horizontal, Angular, EqualWeight)
│   └── conditional_kernel_density/  # CKD non-parametric density estimation
├── data/                  # Data download, NWP preprocessing, training data pipeline
├── trainers/              # Specialized trainers (CKD, Angular, CV, minT)
└── utils/                 # Evaluation metrics (nMAPE, CRPS, grid_crps), visualization
```

## ForecastResult

Every model's `forecast()` returns a **ForecastResult** object. The concrete type and shape depend on the model family and forecasting method.

### Return Shape by Model and Method

#### Statistical Models (RollingRunner — StatefulPredictor)

Recursive forecasting: `forecast()` → `update_state()` loop. Each call produces a single forecast origin.

| Model | Method | Single Call Shape | Runner Aggregation | Final Shape |
|-------|--------|-------------------|--------------------|-------------|
| ARIMA-GARCH | `forecast(horizon)` | `ParametricForecastResult` (1, H) | concat axis=0 | **(N, H)** |
| SARIMA-GARCH | `forecast(horizon)` | `ParametricForecastResult` (1, H) | concat axis=0 | **(N, H)** |
| ARFIMA-GARCH | `forecast(horizon)` | `ParametricForecastResult` (1, H) | concat axis=0 | **(N, H)** |
| GarchBase (all) | `simulate_paths(n_paths, horizon)` | `SampleForecastResult` (1, n_paths, H) | concat axis=0 | **(N, n_paths, H)** |

#### Machine Learning Models (PerHorizonRunner)

Cross-sectional forecasting: independent model per horizon predicts all time points at once.

| Model | Method | Single Call Shape | Runner Aggregation | Final Shape |
|-------|--------|-------------------|--------------------|-------------|
| XGBoost | `forecast(X, idx)` | `ParametricForecastResult` (T, 1) | stack axis=1 | **(N_common, H)** |
| CatBoost | `forecast(X, idx)` | `ParametricForecastResult` (T, 1) | stack axis=1 | **(N_common, H)** |
| NGBoost | `forecast(X, idx)` | `ParametricForecastResult` (T, 1) | stack axis=1 | **(N_common, H)** |
| PGBM | `forecast(X, idx)` | `ParametricForecastResult` (T, 1) | stack axis=1 | **(N_common, H)** |
| LR | `forecast(X, idx)` | `ParametricForecastResult` (T, 1) | stack axis=1 | **(N_common, H)** |

- **T** = number of test observations in horizon-specific CSV (varies per horizon)
- **N_common** = intersection of test indices across all H horizons

#### Deep Time Series Models (RollingRunner — ContextPredictor)

Rolling context window forecasting via NeuralForecast. Loss function determines result type.

| Model | Loss | Method | Single Call Shape | Final Shape |
|-------|------|--------|-------------------|-------------|
| DeepAR | `DistributionLoss` | `predict_from_context()` | `ParametricForecastResult` (1, H) | **(N, H)** |
| DeepAR | `MQLoss` / `IQLoss` | `predict_from_context()` | `QuantileForecastResult` (1, H) | **(N, H)** |
| TFT | `DistributionLoss` | `predict_from_context()` | `ParametricForecastResult` (1, H) | **(N, H)** |
| TFT | `MQLoss` / `IQLoss` | `predict_from_context()` | `QuantileForecastResult` (1, H) | **(N, H)** |

#### Foundation Models (RollingRunner — ContextPredictor)

Rolling context window with Monte Carlo sampling from pretrained models. Supports fine-tuning via `FineTuneStrategy` (FULL, HEAD, LORA).

| Model | Method | Single Call Shape | Final Shape |
|-------|--------|-------------------|-------------|
| Chronos | `predict_from_context()` | `SampleForecastResult` (1, n_samples, H) | **(N, n_samples, H)** |
| Moirai | `predict_from_context()` | `SampleForecastResult` (1, n_samples, H) | **(N, n_samples, H)** |

#### Conditional Kernel Density (CKDRunner — per-horizon)

Non-parametric density estimation via kernel density on a fixed grid.

| Model | Method | Single Call Shape | Final Shape |
|-------|--------|-------------------|-------------|
| CKD | `apply()` | `GridForecastResult` (N, G, 1) | **(N, G, H)** |

- **G** = number of grid bins

### Dimension Reference

| Symbol | Meaning |
|--------|---------|
| **N** | Number of basis times (forecast origins / rolling steps) |
| **H** | Forecast horizon (number of steps ahead) |
| **n_samples** | Number of Monte Carlo / stochastic simulation paths |
| **T** | Number of test observations per horizon (PerHorizonRunner) |
| **N_common** | Intersection of test indices across all horizons (PerHorizonRunner) |
| **G** | Number of grid bins (CKD / GridForecastResult) |

### ForecastResult Types

- **`ParametricForecastResult`** — params: `Dict[str, np.ndarray]`, each value shape matches the result shape above (e.g., Normal: `{"loc": (N, H), "scale": (N, H)}`)
- **`QuantileForecastResult`** — quantiles_data: `Dict[float, np.ndarray]`, each value shape `(N, H)` (e.g., `{0.1: (N, H), 0.5: (N, H), 0.9: (N, H)}`)
- **`SampleForecastResult`** — samples: `np.ndarray` of shape `(N, n_samples, H)`
- **`GridForecastResult`** — grid_probs: `np.ndarray` of shape `(N, G, H)`, bin edges defining the grid

### ForecastResult to Distribution

All ForecastResult objects can be converted to a **Distribution** object for a specific horizon `h`:

```python
result = model.forecast(...)
dist = result.to_distribution(h=6)

# ParametricForecastResult → ParametricDistribution (scipy-backed)
# QuantileForecastResult   → QuantileDistribution (piecewise-linear CDF)
# SampleForecastResult     → SampleDistribution (ECDF-based)
# GridForecastResult       → GridDistribution (histogram-backed)

dist.ppf([0.1, 0.5, 0.9])   # quantiles
dist.mean()                   # point forecast
dist.sample(1000)             # random draws
```

## Runner

Three Runner patterns orchestrate forecasting depending on the model type:

| Runner | Strategy | Models |
|--------|----------|--------|
| `RollingRunner` | Recursive: forecast → update_state (Stateful) or predict_from_context (Context) loop. Each step returns a single origin, stacked along axis=0 to (N, ...). | Statistical, Deep, Foundation |
| `PerHorizonRunner` | Cross-sectional: independent model per horizon. Each model returns (T, 1), stacked along axis=1 to (N_common, H). | Machine Learning |
| `CKDRunner` | Per-horizon CKD orchestration with optional Optuna bandwidth/time-decay optimization. Produces GridForecastResult (N, G, H). | CKD |

## Forecast Combining

Multiple forecasts can be combined into a single probabilistic forecast:

| Combiner | Strategy | Reference |
|----------|----------|-----------|
| `VerticalCombiner` | CDF weighted average (Linear Pool), SLSQP optimization | — |
| `HorizontalCombiner` | Quantile function averaging | — |
| `AngularCombiner` | Interpolation between horizontal (θ=0°) and vertical (θ=90°) | Taylor & Meng 2025 |
| `EqualWeightCombiner` | Equal weight averaging | — |

All combiners consume multiple `ForecastResult` objects and produce `QuantileForecastResult`.

## Setup

```bash
uv sync          # install dependencies
uv run pytest    # run tests
```

- Python 3.13 (managed by `uv`)
- Build backend: hatchling
- Linter: ruff
