"""End-to-end tests: SampleForecastResult + Foundation models.

Covers pretrained foundation models that produce SampleForecastResult
(or QuantileForecastResult when output_type="quantiles") via
BaseFoundationModel's sample-based inference.

Models tested:
    - ChronosForecaster (Amazon Chronos, univariate)
    - MoiraiForecaster  (Salesforce Moirai, supports exogenous)
"""

import numpy as np
import pytest

from src.core.forecast_distribution import SampleDistribution
from src.core.forecast_results import QuantileForecastResult, SampleForecastResult

from .conftest import Y_COL

PREDICTION_LENGTH = 6
N_SAMPLES = 20


# ======================================================================
# Chronos
# ======================================================================

@pytest.mark.slow
class TestChronosForecaster:
    """Chronos: pretrained language-model-based time series forecaster."""

    def test_samples_e2e(self, train_df, tmp_path):
        """fit → forecast → SampleForecastResult (1, n_samples, H)."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": 64,
                "output_type": "samples",
        })
        model.fit(dataset=train_df, y_col=Y_COL)
        assert model.is_fitted_

        result = model.forecast()

        assert isinstance(result, SampleForecastResult)
        assert result.samples.shape == (1, N_SAMPLES, PREDICTION_LENGTH)
        assert np.all(np.isfinite(result.samples))

        # mean / std via Distribution
        dist = result.to_distribution(1)
        assert dist.mean().shape == (1,)
        assert np.all(dist.std() > 0)

    def test_quantile_output(self, train_df, tmp_path):
        """output_type='quantiles' → QuantileForecastResult."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": 64,
                "output_type": "quantiles",
        })
        model.fit(dataset=train_df, y_col=Y_COL)
        result = model.forecast()

        assert isinstance(result, QuantileForecastResult)
        assert 0.5 in result.quantile_levels
        assert result.horizon == PREDICTION_LENGTH

    def test_to_distribution(self, train_df, tmp_path):
        """to_distribution(h) returns SampleDistribution from samples."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": 64,
        })
        model.fit(dataset=train_df, y_col=Y_COL)
        result = model.forecast()

        dist = result.to_distribution(h=1)
        assert isinstance(dist, SampleDistribution)

    def test_predict_from_context(self, train_df, tmp_path):
        """predict_from_context → SampleForecastResult for rolling use."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": 64,
        })
        model.fit(dataset=train_df, y_col=Y_COL)

        context_y = train_df[Y_COL].to_numpy()[-64:]
        result = model.predict_from_context(context_y=context_y, horizon=PREDICTION_LENGTH)

        assert isinstance(result, SampleForecastResult)
        assert result.samples.shape == (1, N_SAMPLES, PREDICTION_LENGTH)


# ======================================================================
# Moirai
# ======================================================================

@pytest.mark.slow
class TestMoiraiForecaster:
    """Moirai: universal time series forecaster with exogenous support."""

    def test_samples_e2e(self, train_df, tmp_path):
        """fit → forecast → SampleForecastResult (1, n_samples, H)."""
        from src.models.foundation.moirai import MoiraiForecaster

        model = MoiraiForecaster(hyperparameter={
                "model_name_or_path": "Salesforce/moirai-1.0-R-small",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": 64,
        })
        model.fit(dataset=train_df, y_col=Y_COL)
        assert model.is_fitted_

        result = model.forecast()

        assert isinstance(result, SampleForecastResult)
        assert result.samples.shape == (1, N_SAMPLES, PREDICTION_LENGTH)
        assert np.all(np.isfinite(result.samples))

    def test_with_exogenous(self, train_df, tmp_path):
        """Moirai with exog_cols (past covariates) produces valid result."""
        from src.models.foundation.moirai import MoiraiForecaster

        model = MoiraiForecaster(hyperparameter={
                "model_name_or_path": "Salesforce/moirai-1.0-R-small",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": 64,
        })
        model.fit(dataset=train_df, y_col=Y_COL, exog_cols=["wind_speed"])
        result = model.forecast()

        assert isinstance(result, SampleForecastResult)
        assert result.samples.shape[2] == PREDICTION_LENGTH

    def test_quantile_output(self, train_df, tmp_path):
        """output_type='quantiles' → QuantileForecastResult."""
        from src.models.foundation.moirai import MoiraiForecaster

        model = MoiraiForecaster(hyperparameter={
                "model_name_or_path": "Salesforce/moirai-1.0-R-small",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": 64,
                "output_type": "quantiles",
        })
        model.fit(dataset=train_df, y_col=Y_COL)
        result = model.forecast()

        assert isinstance(result, QuantileForecastResult)
        assert 0.5 in result.quantile_levels
