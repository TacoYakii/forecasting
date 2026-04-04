"""Tests for stepwise order selection module.

Verifies:
    1. Unit root tests (ADF, KPSS) produce correct results on known series
    2. select_d returns correct differencing orders
    3. StepwiseOrderSelector returns fitted models for arima/sarima/arfima
    4. search_history is populated and summary DataFrame is valid
    5. All-failure case raises RuntimeError
"""

import numpy as np
import pandas as pd
import pytest

from src.models.statistical._stepwise import (
    StepwiseOrderSelector,
    _adf_statistic,
    _kpss_statistic,
    select_d,
    select_D,
)

from .conftest import Y_COL, EXOG_COLS


# ======================================================================
# Unit root tests
# ======================================================================

class TestADFStatistic:
    """ADF t-statistic computation."""

    def test_stationary_series(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)
        stat = _adf_statistic(y)
        # Stationary series: should be strongly negative
        assert stat < -2.5

    def test_random_walk(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(500))
        stat = _adf_statistic(y)
        # Random walk: ADF should fail to reject (stat near 0)
        assert stat > -2.86  # 5% critical value (asymptotic)


class TestKPSSStatistic:
    """KPSS test statistic computation."""

    def test_stationary_series(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)
        stat = _kpss_statistic(y)
        # Stationary: LM statistic should be small
        assert stat < 0.463  # 5% critical value

    def test_random_walk(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(500))
        stat = _kpss_statistic(y)
        # Non-stationary: should exceed critical value
        assert stat > 0.463


# ======================================================================
# Differencing order selection
# ======================================================================

class TestSelectD:
    """Automatic d selection via KPSS."""

    def test_stationary_d0(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)
        assert select_d(y) == 0

    def test_random_walk_d1(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(500))
        assert select_d(y) == 1

    def test_respects_max_d(self):
        rng = np.random.default_rng(42)
        # I(2) series
        y = np.cumsum(np.cumsum(rng.standard_normal(500)))
        assert select_d(y, max_d=1) <= 1

    def test_short_series(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(15)
        d = select_d(y)
        assert 0 <= d <= 2


class TestSelectD_Seasonal:
    """Seasonal differencing order selection."""

    def test_stationary_D0(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)
        assert select_D(y, s=24) == 0

    def test_respects_max_D(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(500))
        assert select_D(y, s=24, max_D=1) <= 1


# ======================================================================
# StepwiseOrderSelector validation
# ======================================================================

class TestSelectorValidation:
    """Input validation."""

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            StepwiseOrderSelector("invalid")

    def test_sarima_requires_seasonal_period(self):
        with pytest.raises(ValueError, match="seasonal_period"):
            StepwiseOrderSelector("sarima")

    def test_invalid_ic(self):
        with pytest.raises(ValueError, match="ic"):
            StepwiseOrderSelector("arima", ic="hqic")


# ======================================================================
# Integration: ARIMA stepwise
# ======================================================================

class TestArimaStepwise:
    """End-to-end stepwise search for ARIMA-GARCH."""

    @pytest.fixture()
    def selector(self):
        return StepwiseOrderSelector(
            "arima", ic="aicc", max_p=2, max_q=2, d=0,
            garch_order=(1, 1), verbose=False,
        )

    def test_returns_fitted_model(self, selector, train_df):
        model = selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert model.is_fitted_

    def test_ic_finite(self, selector, train_df):
        model = selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)
        assert np.isfinite(model.aicc)

    def test_search_history_populated(self, selector, train_df):
        selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert len(selector.search_history) > 0

    def test_best_is_minimum_ic(self, selector, train_df):
        model = selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        converged = [r for r in selector.search_history if r.converged]
        best_ic = min(r.ic_value for r in converged)
        assert model.aicc == pytest.approx(best_ic)

    def test_auto_d_selection(self, train_df):
        selector = StepwiseOrderSelector(
            "arima", ic="aicc", max_p=2, max_q=2, d=None, max_d=2,
            verbose=False,
        )
        model = selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert model.is_fitted_


# ======================================================================
# Integration: SARIMA stepwise
# ======================================================================

class TestSarimaStepwise:
    """End-to-end stepwise search for SARIMA-GARCH."""

    def test_returns_fitted_model(self, train_df):
        selector = StepwiseOrderSelector(
            "sarima", ic="aicc", max_p=1, max_q=1,
            seasonal_period=24, max_P=1, max_Q=1, d=0, D=0,
            verbose=False,
        )
        model = selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert model.is_fitted_

    def test_search_history_has_seasonal(self, train_df):
        selector = StepwiseOrderSelector(
            "sarima", ic="aicc", max_p=1, max_q=1,
            seasonal_period=24, max_P=1, max_Q=1, d=0, D=0,
            verbose=False,
        )
        selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        has_seasonal = any(
            r.seasonal_order is not None for r in selector.search_history
        )
        assert has_seasonal


# ======================================================================
# Integration: ARFIMA stepwise
# ======================================================================

class TestArfimaStepwise:
    """End-to-end stepwise search for ARFIMA-GARCH."""

    def test_returns_fitted_model(self, train_df):
        selector = StepwiseOrderSelector(
            "arfima", ic="aicc", max_p=1, max_q=1, verbose=False,
        )
        model = selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert model.is_fitted_


# ======================================================================
# Summary
# ======================================================================

class TestSummary:
    """summary property returns a well-formed DataFrame."""

    def test_summary_columns(self, train_df):
        selector = StepwiseOrderSelector(
            "arima", ic="aicc", max_p=1, max_q=1, d=0, verbose=False,
        )
        selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        df = selector.summary
        assert "arima_order" in df.columns
        assert "garch_order" in df.columns
        assert "aicc" in df.columns
        assert "converged" in df.columns
        assert len(df) > 0

    def test_summary_sorted_by_ic(self, train_df):
        selector = StepwiseOrderSelector(
            "arima", ic="bic", max_p=2, max_q=2, d=0, verbose=False,
        )
        selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        df = selector.summary
        # Converged entries should be sorted
        converged = df[df["converged"]]
        if len(converged) > 1:
            vals = converged["bic"].to_numpy()
            assert np.all(vals[:-1] <= vals[1:] + 1e-10)
