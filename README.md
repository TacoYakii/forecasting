# Windpower Forecasting

A Python 3.13 probabilistic wind power forecasting system.

Provides a unified interface across multiple model families (statistical, machine learning, deep learning, foundation models). Every model returns a probabilistic forecast via the `ForecastResult` abstraction.

## Architecture

```
src/
├── core/                  # Core abstractions (BaseModel, ForecastResult, Registry, Runner)
├── models/
│   ├── statistical/       # ARIMA-GARCH, SARIMA-GARCH, ARFIMA-GARCH
│   ├── machine_learning/  # XGBoost, CatBoost, NGBoost, PGBM, LR, GBM
│   ├── deep_time_series/  # DeepAR, TFT (NeuralForecast)
│   └── foundation/        # Chronos, Moirai (pretrained)
├── data/                  # Data download and preprocessing pipeline
├── pipelines/             # Hierarchical forecasting
└── utils/                 # Evaluation metrics (nMAPE, CRPS), hyperparameter optimization
```

## ForecastResult

Every model's `forecast()` returns a **ForecastResult** object. The concrete type depends on the model's training objective:

```
MLE-based:      Model.forecast() → ParametricForecastResult  (N, H)
Quantile-based: Model.forecast() → QuantileForecastResult    (N, H)
Sample-based:   Model.forecast() → SampleForecastResult      (N, n_samples, H)
```

### ParametricForecastResult

The model assumes a parametric distribution and estimates its native parameters (e.g., mu/sigma for Normal, mu/sigma/df for Student-t). The **Distribution Registry** maps distribution names to scipy distributions and parameter names.

- Statistical models: ARIMA-GARCH, SARIMA-GARCH, ARFIMA-GARCH
- ML models: NGBoost, CatBoost, PGBM, XGBoost, LR, GBM
- Deep models with `DistributionLoss`: DeepAR, TFT

### QuantileForecastResult

The model directly predicts values at specific quantile levels without assuming a distributional form.

- Deep models with `MQLoss` / `IQLoss`: DeepAR, TFT

### SampleForecastResult

The model generates draws (sample paths) from the predictive distribution, representing it empirically. Covers both Monte Carlo sampling from generative models and forward simulation of stochastic processes.

- Foundation models: Chronos, Moirai (Monte Carlo sampling)
- GARCH family: simulated sample paths via `simulate_paths()`

### ForecastResult to Distribution

All ForecastResult objects can be converted to a **Distribution** object for a specific horizon `h`:

```python
result = model.forecast(...)
dist = result.to_distribution(h=6)

# ParametricForecastResult → ParametricDistribution (scipy-backed)
# QuantileForecastResult   → EmpiricalDistribution (interpolated)
# SampleForecastResult     → EmpiricalDistribution (from samples)

dist.ppf([0.1, 0.5, 0.9])   # quantiles
dist.mean()                   # point forecast
dist.crps(observed)           # CRPS score
```

## Runner

Two Runner patterns orchestrate forecasting depending on the model type:

| Runner | Strategy | Models |
|--------|----------|--------|
| `RollingRunner` | Recursive: forecast → update_state loop. Each step returns (1, H), stacked to (N, H). | Statistical, Deep, Foundation |
| `PerHorizonRunner` | Cross-sectional: independent model per horizon. Each model returns (N, 1), stacked to (N, H). | Machine Learning |

## Setup

```bash
uv sync          # install dependencies
uv run pytest    # run tests
```

- Python 3.13 (managed by `uv`)
- Build backend: hatchling
- Linter: ruff
