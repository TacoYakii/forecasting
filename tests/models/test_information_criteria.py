"""Tests for joint ARMA-GARCH information criteria (AIC, BIC, AICc, loglik).

Verifies that:
    1. Properties exist and return finite values after fit
    2. AIC/BIC/AICc relationships hold (AIC < BIC for large n, AICc > AIC)
    3. More parameters → higher penalty (AIC comparison between models)
    4. studentT vs normal: loglik differs, df adds 1 to n_params
    5. Properties raise RuntimeError before fit
"""

import numpy as np
import pytest

from src.models.statistical.arima_garch import ArimaGarchForecaster
from src.models.statistical.sarima_garch import SarimaGarchForecaster
from src.models.statistical.arfima_garch import ArfimaGarchForecaster

from .conftest import Y_COL, EXOG_COLS, TRAIN_END


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture()
def fitted_arima(train_df):
    """Fitted ARIMA(1,0,1)-GARCH(1,1) normal."""
    model = ArimaGarchForecaster(hyperparameter={
        "arima_order": (1, 0, 1),
        "garch_order": (1, 1),
        "distribution": "normal",
    })
    model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
    return model


@pytest.fixture()
def fitted_arima_t(train_df):
    """Fitted ARIMA(1,0,1)-GARCH(1,1) studentT."""
    model = ArimaGarchForecaster(hyperparameter={
        "arima_order": (1, 0, 1),
        "garch_order": (1, 1),
        "distribution": "studentT",
    })
    model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
    return model


# ======================================================================
# Tests
# ======================================================================

class TestICProperties:
    """Basic property access and sanity checks."""

    def test_aic_finite(self, fitted_arima):
        assert np.isfinite(fitted_arima.aic)

    def test_bic_finite(self, fitted_arima):
        assert np.isfinite(fitted_arima.bic)

    def test_aicc_finite(self, fitted_arima):
        assert np.isfinite(fitted_arima.aicc)

    def test_loglik_finite(self, fitted_arima):
        assert np.isfinite(fitted_arima.loglik)

    def test_loglik_negative(self, fitted_arima):
        # Log-likelihood is typically negative for continuous distributions
        assert fitted_arima.loglik < 0

    def test_aic_bic_relationship(self, fitted_arima):
        # For n >> k, ln(n) > 2, so BIC > AIC
        assert fitted_arima.bic > fitted_arima.aic

    def test_aicc_greater_than_aic(self, fitted_arima):
        # AICc = AIC + correction (always positive)
        assert fitted_arima.aicc > fitted_arima.aic

    def test_aic_formula(self, fitted_arima):
        # AIC = 2*NLL + 2*k
        expected = -2.0 * fitted_arima.loglik + 2.0 * fitted_arima._n_params
        assert fitted_arima.aic == pytest.approx(expected)

    def test_bic_formula(self, fitted_arima):
        # BIC = 2*NLL + k*ln(n)
        expected = (-2.0 * fitted_arima.loglik
                    + fitted_arima._n_params * np.log(fitted_arima._n_obs))
        assert fitted_arima.bic == pytest.approx(expected)


class TestICBeforeFit:
    """Properties must raise before fit."""

    def test_aic_raises(self):
        model = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1),
            "garch_order": (1, 1),
            "distribution": "normal",
        })
        with pytest.raises(RuntimeError):
            _ = model.aic

    def test_bic_raises(self):
        model = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1),
            "garch_order": (1, 1),
            "distribution": "normal",
        })
        with pytest.raises(RuntimeError):
            _ = model.bic


class TestICDistribution:
    """studentT vs normal comparison."""

    def test_student_t_extra_param(self, fitted_arima, fitted_arima_t):
        # studentT has 1 extra parameter (df)
        assert fitted_arima_t._n_params == fitted_arima._n_params + 1

    def test_same_n_obs(self, fitted_arima, fitted_arima_t):
        assert fitted_arima_t._n_obs == fitted_arima._n_obs

    def test_loglik_differs(self, fitted_arima, fitted_arima_t):
        # Different distributions → different log-likelihood
        assert fitted_arima.loglik != pytest.approx(fitted_arima_t.loglik)


class TestICModelComparison:
    """Compare IC across different model specifications."""

    def test_param_count_increases_with_order(self, train_df):
        small = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 0),
            "garch_order": (1, 1),
            "distribution": "normal",
        })
        large = ArimaGarchForecaster(hyperparameter={
            "arima_order": (2, 0, 2),
            "garch_order": (1, 1),
            "distribution": "normal",
        })
        small.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        large.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)

        # (2,0,2) has 3 more params than (1,0,0)
        assert large._n_params == small._n_params + 3

    def test_sarima_has_more_params(self, train_df):
        arima = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1),
            "garch_order": (1, 1),
            "distribution": "normal",
        })
        sarima = SarimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1),
            "seasonal_order": (1, 0, 1, 24),
            "garch_order": (1, 1),
            "distribution": "normal",
        })
        arima.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        sarima.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)

        # SARIMA adds seasonal AR + seasonal MA params
        assert sarima._n_params > arima._n_params
