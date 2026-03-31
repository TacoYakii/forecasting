"""End-to-end tests: ParametricForecastResult + RollingRunner.

Covers statistical GARCH-family models that produce ParametricForecastResult
via stateful rolling forecasting (StatefulPredictor protocol).

Models tested:
    - ArimaGarchForecaster
    - SarimaGarchForecaster
    - ArfimaGarchForecaster
    - RollingRunner with StatefulPredictor
    - simulate_paths → SampleForecastResult
"""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import ParametricForecastResult, SampleForecastResult
from src.core.forecast_distribution import ParametricDistribution

from .conftest import Y_COL, EXOG_COLS, TRAIN_END, FORECAST_START, FORECAST_END

HORIZON = 6


# ======================================================================
# Helpers
# ======================================================================

def _make_config(model_key: str):
    """Create a minimal config for a GARCH-family model."""
    if model_key == "arima_garch":
        from src.models.statistical.config import ArimaGarchConfig
        return ArimaGarchConfig(
            arima_order=(1, 0, 1),
            garch_order=(1, 1),
            distribution="normal",
        )
    elif model_key == "sarima_garch":
        from src.models.statistical.config import SarimaGarchConfig
        return SarimaGarchConfig(
            arima_order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            garch_order=(1, 1),
            distribution="normal",
        )
    elif model_key == "arfima_garch":
        from src.models.statistical.config import ArfimaGarchConfig
        return ArfimaGarchConfig(
            arfima_order=(1, 1),
            garch_order=(1, 1),
            distribution="normal",
            truncation_K=100,
        )
    raise ValueError(f"Unknown model key: {model_key}")


def _make_model(model_key: str, train_df, tmp_path, exog_cols=None):
    """Instantiate a GARCH-family model."""
    config = _make_config(model_key)

    if model_key == "arima_garch":
        from src.models.statistical.arima_garch import ArimaGarchForecaster
        cls = ArimaGarchForecaster
    elif model_key == "sarima_garch":
        from src.models.statistical.sarima_garch import SarimaGarchForecaster
        cls = SarimaGarchForecaster
    elif model_key == "arfima_garch":
        from src.models.statistical.arfima_garch import ArfimaGarchForecaster
        cls = ArfimaGarchForecaster
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return cls(
        dataset=train_df,
        y_col=Y_COL,
        exog_cols=exog_cols,
        config=config,
        enable_logging=False,
        save_dir=str(tmp_path),
    )


# ======================================================================
# Individual model: fit → forecast → ParametricForecastResult
# ======================================================================

class TestGarchForecasters:
    """GARCH-family: single-shot forecast from fitted state."""

    @pytest.fixture(params=["arima_garch", "sarima_garch", "arfima_garch"])
    def model_key(self, request):
        return request.param

    def test_fit_forecast_e2e(self, model_key, train_df, tmp_path):
        """fit → forecast(horizon) → ParametricForecastResult (1, H)."""
        model = _make_model(model_key, train_df, tmp_path)
        model.fit()
        assert model.is_fitted_

        result = model.forecast(horizon=HORIZON)

        assert isinstance(result, ParametricForecastResult)
        assert result.params["loc"].shape == (1, HORIZON)
        assert result.params["scale"].shape == (1, HORIZON)
        assert np.all(np.isfinite(result.mean()))
        assert np.all(result.std() > 0)

    def test_to_distribution(self, train_df, tmp_path):
        """to_distribution(h) returns ParametricDistribution."""
        model = _make_model("arima_garch", train_df, tmp_path)
        model.fit()
        result = model.forecast(horizon=HORIZON)

        for h in range(1, HORIZON + 1):
            dist = result.to_distribution(h)
            assert isinstance(dist, ParametricDistribution)

    def test_update_state(self, train_df, forecast_df, tmp_path):
        """update_state advances internal state without error."""
        model = _make_model("arima_garch", train_df, tmp_path)
        model.fit()

        y_actual = forecast_df[Y_COL].iloc[0]
        model.update_state(y_actual)

        # Should still produce valid forecast after state update
        result = model.forecast(horizon=HORIZON)
        assert isinstance(result, ParametricForecastResult)
        assert np.all(np.isfinite(result.mean()))

    def test_student_t_distribution(self, train_df, tmp_path):
        """Student-t distribution produces 3-param result (loc, scale, df)."""
        from src.models.statistical.arima_garch import ArimaGarchForecaster
        from src.models.statistical.config import ArimaGarchConfig

        config = ArimaGarchConfig(
            arima_order=(1, 0, 1),
            garch_order=(1, 1),
            distribution="studentT",
        )
        model = ArimaGarchForecaster(
            dataset=train_df, y_col=Y_COL,
            config=config, enable_logging=False, save_dir=str(tmp_path),
        )
        model.fit()
        result = model.forecast(horizon=HORIZON)

        assert result.dist_name == "studentT"
        assert "df" in result.params
        assert result.params["df"].shape == (1, HORIZON)


# ======================================================================
# simulate_paths → SampleForecastResult
# ======================================================================

class TestSimulatePaths:
    """GARCH simulate_paths produces SampleForecastResult."""

    def test_simulate_paths_e2e(self, train_df, tmp_path):
        """simulate_paths → SampleForecastResult (1, n_paths, H)."""
        model = _make_model("arima_garch", train_df, tmp_path)
        model.fit()

        n_paths = 50
        result = model.simulate_paths(n_paths=n_paths, horizon=HORIZON, seed=42)

        assert isinstance(result, SampleForecastResult)
        assert result.samples.shape == (1, n_paths, HORIZON)
        assert np.all(np.isfinite(result.samples))

        # Quantile extraction
        q90 = result.quantile(0.9, h=1)
        assert q90.shape == (1,)
        assert np.isfinite(q90[0])


# ======================================================================
# RollingRunner: stateful rolling evaluation
# ======================================================================

class TestRollingRunner:
    """RollingRunner with StatefulPredictor (GARCH family)."""

    def test_rolling_forecast_e2e(self, full_df, tmp_path):
        """Rolling forecast → stacked ParametricForecastResult (N, H)."""
        from src.core.runner import RollingRunner

        # Train on data up to TRAIN_END
        train_df = full_df.loc[:TRAIN_END]
        model = _make_model("arima_garch", train_df, tmp_path)
        model.fit()

        runner = RollingRunner(
            model=model,
            dataset=full_df,
            y_col=Y_COL,
            forecast_period=(FORECAST_START, FORECAST_END),
        )

        result = runner.run(
            horizon=HORIZON,
            dist_name="normal",
            show_progress=False,
        )

        assert isinstance(result, ParametricForecastResult)

        N = len(full_df.loc[FORECAST_START:FORECAST_END])
        assert result.params["loc"].shape[0] == N
        assert result.params["loc"].shape[1] == HORIZON
        assert np.all(np.isfinite(result.mean()))

    def test_rolling_simulate_paths(self, full_df, tmp_path):
        """Rolling simulate_paths → stacked SampleForecastResult (N, n_paths, H)."""
        from src.core.runner import RollingRunner

        train_df = full_df.loc[:TRAIN_END]
        model = _make_model("arima_garch", train_df, tmp_path)
        model.fit()

        # Use a short forecast period for speed
        short_end = pd.Timestamp(FORECAST_START) + pd.Timedelta(hours=5)
        runner = RollingRunner(
            model=model,
            dataset=full_df,
            y_col=Y_COL,
            forecast_period=(FORECAST_START, str(short_end)),
        )

        n_paths = 30
        result = runner.run(
            horizon=HORIZON,
            method="simulate_paths",
            method_kwargs={"n_paths": n_paths, "seed": 42},
            show_progress=False,
        )

        assert isinstance(result, SampleForecastResult)
        N = len(full_df.loc[FORECAST_START:str(short_end)])
        assert result.samples.shape == (N, n_paths, HORIZON)
