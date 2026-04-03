"""End-to-end tests: ParametricForecastResult + PerHorizonRunner.

Covers ML models that produce ParametricForecastResult via the
per-horizon cross-sectional forecasting pattern.

Models tested:
    - LRForecaster, XGBoostForecaster, CatBoostForecaster (DeterministicForecaster)
    - NGBoostForecaster, PGBMForecaster (native probabilistic)
    - PerHorizonRunner (multi-horizon orchestration)
"""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import ParametricForecastResult
from src.core.forecast_distribution import ParametricDistribution

from .conftest import Y_COL, EXOG_COLS, TRAIN_END, FORECAST_START, FORECAST_END

# Trigger MODEL_REGISTRY registration
import importlib as _il
for _mod in [
    "src.models.machine_learning.lr_model",
    "src.models.machine_learning.xgboost_model",
    "src.models.machine_learning.catboost_model",
    "src.models.machine_learning.gboost_model",
    "src.models.machine_learning.ngboost_model",
    "src.models.machine_learning.pgboost_model",
]:
    try:
        _il.import_module(_mod)
    except Exception:
        pass


# ======================================================================
# Individual model: fit → forecast → ParametricForecastResult
# ======================================================================

class TestDeterministicForecasters:
    """DeterministicForecaster subclasses (moment-matching based)."""

    @pytest.fixture(params=["lr", "xgboost", "catboost", "gbm"])
    def model_name(self, request):
        return request.param

    def test_fit_forecast_e2e(self, model_name, train_df, forecast_df, tmp_path):
        """fit → forecast → ParametricForecastResult with correct shape."""
        from src.core.registry import MODEL_REGISTRY

        model_cls = MODEL_REGISTRY.get(model_name)
        model = model_cls()
        model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert model.is_fitted_

        forecast_X = forecast_df[EXOG_COLS].to_numpy()
        forecast_index = forecast_df.index
        result = model.forecast(forecast_X, forecast_index)

        # Type check
        assert isinstance(result, ParametricForecastResult)

        # Shape: (T, 1) — single horizon per model
        T = len(forecast_index)
        assert result.params["loc"].shape == (T, 1)
        assert result.params["scale"].shape == (T, 1)
        assert len(result.basis_index) == T

        # Values are finite (check via Distribution)
        dist = result.to_distribution(1)
        assert np.all(np.isfinite(dist.mean()))
        assert np.all(np.isfinite(dist.std()))
        assert np.all(dist.std() > 0)

    def test_to_distribution(self, train_df, forecast_df, tmp_path):
        """to_distribution(h=1) returns a usable ParametricDistribution."""
        from src.models.machine_learning.lr_model import LRForecaster

        model = LRForecaster()
        model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)

        forecast_X = forecast_df[EXOG_COLS].to_numpy()
        result = model.forecast(forecast_X, forecast_df.index)

        dist = result.to_distribution(h=1)
        assert isinstance(dist, ParametricDistribution)

        # ppf should return finite values
        quantiles = dist.ppf([0.1, 0.5, 0.9])
        assert quantiles.shape == (len(forecast_df), 3)
        assert np.all(np.isfinite(quantiles))

    def test_to_dataframe(self, train_df, forecast_df, tmp_path):
        """to_dataframe() returns a well-formed DataFrame."""
        from src.models.machine_learning.lr_model import LRForecaster

        model = LRForecaster()
        model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)

        result = model.forecast(forecast_df[EXOG_COLS].to_numpy(), forecast_df.index)
        df_out = result.to_dataframe(h=1)

        assert "mu" in df_out.columns
        assert "std" in df_out.columns
        assert len(df_out) == len(forecast_df)


class TestNativeProbabilisticForecasters:
    """NGBoost / PGBM — native probabilistic models (not DeterministicForecaster)."""

    @pytest.fixture(params=["ngboost", "pgbm"])
    def model_name(self, request):
        return request.param

    def test_fit_forecast_e2e(self, model_name, train_df, forecast_df, tmp_path):
        """fit → forecast → ParametricForecastResult with native params."""
        from src.core.registry import MODEL_REGISTRY

        model_cls = MODEL_REGISTRY.get(model_name)
        model = model_cls()
        model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert model.is_fitted_

        forecast_X = forecast_df[EXOG_COLS].to_numpy()
        result = model.forecast(forecast_X, forecast_df.index)

        assert isinstance(result, ParametricForecastResult)
        T = len(forecast_df)
        for v in result.params.values():
            assert v.shape == (T, 1)
        assert np.all(np.isfinite(result.to_distribution(1).mean()))


# ======================================================================
# PerHorizonRunner: multi-horizon orchestration
# ======================================================================

class TestPerHorizonRunner:
    """PerHorizonRunner: train H models → unified (N, H) result."""

    def test_runner_fit_forecast_e2e(self, per_horizon_csv_dir, tmp_path):
        """Full e2e: CSV load → fit → forecast → (N, H) ParametricForecastResult."""
        from src.core.runner import PerHorizonRunner

        runner = PerHorizonRunner(
            data_dir=per_horizon_csv_dir,
            registry_key="lr",
            y_col=Y_COL,
            exog_cols=EXOG_COLS,
            training_period=("2023-01-01", TRAIN_END),
            forecast_period=(FORECAST_START, FORECAST_END),
            save_dir=str(tmp_path),
        )
        runner.fit()
        assert runner.is_fitted_

        result = runner.forecast()

        assert isinstance(result, ParametricForecastResult)

        H = len(runner.horizons)  # 3
        N = len(result.basis_index)
        assert H == 3
        assert result.params["loc"].shape == (N, H)
        assert result.params["scale"].shape == (N, H)
        assert N > 0

        # to_distribution for each horizon
        for h in range(1, H + 1):
            dist = result.to_distribution(h)
            assert isinstance(dist, ParametricDistribution)

    def test_runner_single_horizon(self, per_horizon_csv_dir, tmp_path):
        """forecast_horizon(h) returns (T, 1) for a single horizon."""
        from src.core.runner import PerHorizonRunner

        runner = PerHorizonRunner(
            data_dir=per_horizon_csv_dir,
            registry_key="lr",
            y_col=Y_COL,
            exog_cols=EXOG_COLS,
            training_period=("2023-01-01", TRAIN_END),
            forecast_period=(FORECAST_START, FORECAST_END),
            save_dir=str(tmp_path),
        )
        runner.fit()
        result_h1 = runner.forecast_horizon(1)

        assert isinstance(result_h1, ParametricForecastResult)
        assert result_h1.horizon == 1
