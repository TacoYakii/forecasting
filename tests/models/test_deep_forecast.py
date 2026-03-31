"""End-to-end tests: Deep learning models via NeuralForecast.

Covers BaseDeepModel subclasses that produce ParametricForecastResult
(DistributionLoss) or QuantileForecastResult (MQLoss/IQLoss).

Models tested:
    - DeepARForecaster  (default: DistributionLoss → ParametricForecastResult)
    - TFTForecaster     (default: MQLoss → QuantileForecastResult)
    - Loss type switching (distribution ↔ quantile)
"""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import ParametricForecastResult, QuantileForecastResult

from .conftest import Y_COL, EXOG_COLS, TRAIN_END, FORECAST_START

PREDICTION_LENGTH = 6

# Minimal training config for fast tests
_FAST_HP = {
    "prediction_length": PREDICTION_LENGTH,
    "max_steps": 5,
    "batch_size": 16,
    "input_size": 24,
}


@pytest.fixture()
def future_exog(synthetic_df):
    """Future exogenous features for the forecast horizon."""
    future_df = synthetic_df.loc[FORECAST_START:]
    future_X = future_df[EXOG_COLS].to_numpy()[:PREDICTION_LENGTH]
    future_index = future_df.index[:PREDICTION_LENGTH]
    return future_X, future_index


# ======================================================================
# DeepAR
# ======================================================================

@pytest.mark.slow
class TestDeepARForecaster:
    """DeepAR: autoregressive RNN-based probabilistic forecaster."""

    def test_distribution_loss_e2e(self, train_df, future_exog, tmp_path):
        """DistributionLoss → ParametricForecastResult (1, H)."""
        from src.models.deep_time_series.deepar import DeepARForecaster

        hp = {**_FAST_HP, "loss_type": "distribution", "distribution": "Normal"}
        model = DeepARForecaster(
            dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS,
            hyperparameter=hp,
            enable_logging=False, save_dir=str(tmp_path),
        )
        model.fit()
        assert model.is_fitted_

        future_X, future_index = future_exog
        result = model.forecast(future_X=future_X, future_index=future_index)
        assert isinstance(result, ParametricForecastResult)
        assert result.params["loc"].shape == (1, PREDICTION_LENGTH)
        assert np.all(np.isfinite(result.to_distribution(1).mean()))

    def test_quantile_loss_e2e(self, train_df, future_exog, tmp_path):
        """MQLoss → QuantileForecastResult (1, H)."""
        from src.models.deep_time_series.deepar import DeepARForecaster

        hp = {**_FAST_HP, "loss_type": "quantile"}
        model = DeepARForecaster(
            dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS,
            hyperparameter=hp,
            enable_logging=False, save_dir=str(tmp_path),
        )
        model.fit()

        future_X, future_index = future_exog
        result = model.forecast(future_X=future_X, future_index=future_index)

        assert isinstance(result, QuantileForecastResult)
        assert result.horizon == PREDICTION_LENGTH
        assert len(result.quantile_levels) > 0
        assert np.all(np.isfinite(result.to_distribution(1).mean()))

    def test_to_distribution(self, train_df, future_exog, tmp_path):
        """to_distribution returns correct type per loss."""
        from src.models.deep_time_series.deepar import DeepARForecaster
        from src.core.forecast_distribution import EmpiricalDistribution

        hp = {**_FAST_HP, "loss_type": "quantile"}
        model = DeepARForecaster(
            dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS,
            hyperparameter=hp,
            enable_logging=False, save_dir=str(tmp_path),
        )
        model.fit()

        future_X, future_index = future_exog
        result = model.forecast(future_X=future_X, future_index=future_index)

        dist = result.to_distribution(h=1)
        assert isinstance(dist, EmpiricalDistribution)


# ======================================================================
# TFT
# ======================================================================

@pytest.mark.slow
class TestTFTForecaster:
    """TFT: attention-based multi-horizon forecaster."""

    def test_quantile_loss_e2e(self, train_df, future_exog, tmp_path):
        """Default MQLoss → QuantileForecastResult (1, H)."""
        from src.models.deep_time_series.tft import TFTForecaster

        model = TFTForecaster(
            dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS,
            hyperparameter=_FAST_HP,
            enable_logging=False, save_dir=str(tmp_path),
        )
        model.fit()
        assert model.is_fitted_

        future_X, future_index = future_exog
        result = model.forecast(future_X=future_X, future_index=future_index)
        assert isinstance(result, QuantileForecastResult)
        assert result.horizon == PREDICTION_LENGTH
        assert np.all(np.isfinite(result.to_distribution(1).mean()))

    def test_distribution_loss_e2e(self, train_df, future_exog, tmp_path):
        """DistributionLoss → ParametricForecastResult (1, H)."""
        from src.models.deep_time_series.tft import TFTForecaster

        hp = {**_FAST_HP, "loss_type": "distribution", "distribution": "Normal"}
        model = TFTForecaster(
            dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS,
            hyperparameter=hp,
            enable_logging=False, save_dir=str(tmp_path),
        )
        model.fit()

        future_X, future_index = future_exog
        result = model.forecast(future_X=future_X, future_index=future_index)

        assert isinstance(result, ParametricForecastResult)
        assert result.params["loc"].shape == (1, PREDICTION_LENGTH)

    def test_to_dataframe(self, train_df, future_exog, tmp_path):
        """to_dataframe() works for QuantileForecastResult."""
        from src.models.deep_time_series.tft import TFTForecaster

        model = TFTForecaster(
            dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS,
            hyperparameter=_FAST_HP,
            enable_logging=False, save_dir=str(tmp_path),
        )
        model.fit()

        future_X, future_index = future_exog
        result = model.forecast(future_X=future_X, future_index=future_index)

        df_out = result.to_dataframe(h=1)
        assert "mu" in df_out.columns
        assert "std" in df_out.columns
        assert len(df_out) == 1
