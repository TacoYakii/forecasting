"""Tests for the innovation distribution dispatch system.

Verifies:
    - Registry: registration, lookup, unknown distribution error
    - Fitting: normal vs studentT vs skewStudentT produce correct dist_params
    - Forecast: output param keys match distribution
    - Shocks: unit variance, heavier tails for studentT/skewStudentT
    - Serialisation: dist_params preserved through save/load
"""

import numpy as np
import pytest
from scipy.stats import kurtosis

from src.models.statistical._innovations import (
    INNOVATION_REGISTRY,
    NormalInnovation,
    SkewStudentTInnovation,
    StudentTInnovation,
    get_innovation,
)
from src.models.statistical.arima_garch import ArimaGarchForecaster

from .conftest import EXOG_COLS, TRAIN_END, Y_COL

HORIZON = 6


# ======================================================================
# Helpers
# ======================================================================

def _fit_model(train_df, distribution="normal"):
    """Fit a simple ARIMA(1,0,1)-GARCH(1,1) with the given distribution."""
    model = ArimaGarchForecaster(hyperparameter={
        "arima_order": (1, 0, 1),
        "garch_order": (1, 1),
        "distribution": distribution,
    })
    model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
    return model


# ======================================================================
# Registry
# ======================================================================

class TestRegistry:
    """Innovation registry tests."""

    def test_contains_all_distributions(self):
        assert "normal" in INNOVATION_REGISTRY
        assert "studentT" in INNOVATION_REGISTRY
        assert "skewStudentT" in INNOVATION_REGISTRY

    def test_get_innovation_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown innovation"):
            get_innovation("cauchy")

    def test_registry_keys_match_name(self):
        for key, innov in INNOVATION_REGISTRY.items():
            assert key == innov.name


# ======================================================================
# Fitting correctness
# ======================================================================

class TestFittingDistParams:
    """Distribution parameters are correctly stored after fit."""

    def test_normal_no_df(self, train_df):
        model = _fit_model(train_df, "normal")
        assert model._dist_params == {}
        assert model._df is None

    def test_studentt_has_df(self, train_df):
        model = _fit_model(train_df, "studentT")
        assert "df" in model._dist_params
        df = model._dist_params["df"]
        assert 2.0 < df < 100.0
        assert model._df == pytest.approx(df)

    def test_loglik_uses_correct_distribution(self, train_df):
        normal = _fit_model(train_df, "normal")
        student = _fit_model(train_df, "studentT")
        assert normal.loglik != pytest.approx(student.loglik)


# ======================================================================
# Forecast output shape
# ======================================================================

class TestForecastParams:
    """Forecast ParametricForecastResult has correct param keys."""

    def test_normal_forecast_has_loc_scale_only(self, train_df):
        model = _fit_model(train_df, "normal")
        result = model.forecast(horizon=HORIZON)
        assert set(result.params.keys()) == {"loc", "scale"}

    def test_studentt_forecast_has_df(self, train_df):
        model = _fit_model(train_df, "studentT")
        result = model.forecast(horizon=HORIZON)
        assert "df" in result.params
        assert "loc" in result.params
        assert "scale" in result.params

    def test_studentt_scale_adjusted(self, train_df):
        model = _fit_model(train_df, "studentT")
        result = model.forecast(horizon=HORIZON)
        df = model._dist_params["df"]
        # scale should be sigma * sqrt((df-2)/df), which is < sigma
        # loc and scale must be (1, H)
        assert result.params["scale"].shape == (1, HORIZON)
        # All scale values should be positive
        assert np.all(result.params["scale"] > 0)


# ======================================================================
# Shock sampling
# ======================================================================

class TestShockSampling:
    """Standardised shocks have unit variance and correct tail behaviour."""

    N_SAMPLES = 100_000

    def test_normal_shocks_unit_variance(self):
        innov = NormalInnovation()
        rng = np.random.default_rng(42)
        shocks = innov.sample_shocks(rng, (self.N_SAMPLES,), {})
        assert np.var(shocks) == pytest.approx(1.0, abs=0.02)

    def test_studentt_shocks_unit_variance(self):
        innov = StudentTInnovation()
        rng = np.random.default_rng(42)
        shocks = innov.sample_shocks(rng, (self.N_SAMPLES,), {"df": 5.0})
        assert np.var(shocks) == pytest.approx(1.0, abs=0.05)

    def test_studentt_shocks_heavier_tails(self):
        rng_n = np.random.default_rng(42)
        rng_t = np.random.default_rng(42)
        normal_shocks = NormalInnovation().sample_shocks(
            rng_n, (self.N_SAMPLES,), {},
        )
        studentt_shocks = StudentTInnovation().sample_shocks(
            rng_t, (self.N_SAMPLES,), {"df": 5.0},
        )
        assert kurtosis(studentt_shocks) > kurtosis(normal_shocks)


# ======================================================================
# Serialisation round-trip
# ======================================================================

class TestSerialisation:
    """dist_params preserved through save/load."""

    def test_save_load_preserves_dist_params(self, train_df, tmp_path):
        model = _fit_model(train_df, "studentT")
        original_dist_params = model._dist_params.copy()
        original_df = model._df

        # Save
        sv_path = model._save_model_specific(tmp_path / "test_model")

        # Create fresh model and load
        model2 = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1),
            "garch_order": (1, 1),
            "distribution": "studentT",
        })
        model2._load_model_specific(tmp_path / "test_model")

        assert model2._dist_params == original_dist_params
        assert model2._df == pytest.approx(original_df)


# ======================================================================
# Skewed Student-t specific
# ======================================================================

class TestSkewStudentT:
    """Hansen's skewed Student-t fitting, forecast, and shock tests."""

    def test_fitting_has_df_and_skew(self, train_df):
        model = _fit_model(train_df, "skewStudentT")
        assert "df" in model._dist_params
        assert "skew" in model._dist_params
        assert 2.0 < model._dist_params["df"] < 100.0
        assert -1.0 < model._dist_params["skew"] < 1.0

    def test_n_params_two_more_than_normal(self, train_df):
        normal = _fit_model(train_df, "normal")
        skew_t = _fit_model(train_df, "skewStudentT")
        assert skew_t._n_params == normal._n_params + 2

    def test_forecast_has_skew_key(self, train_df):
        model = _fit_model(train_df, "skewStudentT")
        result = model.forecast(horizon=HORIZON)
        assert set(result.params.keys()) == {"loc", "scale", "df", "skew"}

    def test_to_distribution_works(self, train_df):
        model = _fit_model(train_df, "skewStudentT")
        result = model.forecast(horizon=HORIZON)
        dist = result.to_distribution(1)
        median = dist.ppf(0.5)
        assert np.isfinite(median).all()

    def test_shocks_unit_variance(self):
        innov = SkewStudentTInnovation()
        rng = np.random.default_rng(42)
        shocks = innov.sample_shocks(
            rng, (100_000,), {"df": 5.0, "skew": -0.3},
        )
        assert np.var(shocks) == pytest.approx(1.0, abs=0.05)

    def test_shocks_are_skewed(self):
        from scipy.stats import skew as scipy_skew
        innov = SkewStudentTInnovation()
        rng = np.random.default_rng(42)
        shocks = innov.sample_shocks(
            rng, (100_000,), {"df": 5.0, "skew": -0.5},
        )
        # Negative skew parameter should produce negatively skewed samples
        assert scipy_skew(shocks) < 0

    def test_save_load_preserves_skew_params(self, train_df, tmp_path):
        model = _fit_model(train_df, "skewStudentT")
        original = model._dist_params.copy()

        model._save_model_specific(tmp_path / "skew_model")

        model2 = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1),
            "garch_order": (1, 1),
            "distribution": "skewStudentT",
        })
        model2._load_model_specific(tmp_path / "skew_model")

        assert model2._dist_params["df"] == pytest.approx(original["df"])
        assert model2._dist_params["skew"] == pytest.approx(original["skew"])
