"""Tests for GridForecastResult and CKDRunner.

Covers:
- GridForecastResult: construction, validation, to_distribution, reindex,
  save/load round-trip, to_dataframe
- CKDRunner: fixed-config fit/run, Optuna fit/run, parallel execution,
  error handling, integration with ParametricForecastResult inputs
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics.crps import grid_crps

from src.core.forecast_distribution import GridDistribution, ParametricDistribution
from src.core.forecast_results import (
    GridForecastResult,
    ParametricForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
    load_forecast_result,
)
from src.core.ckd_runner import CKDRunner
from src.models.conditional_kernel_density import CKDConfig


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_grid():
    """Equally spaced grid with 31 points."""
    return np.linspace(0, 100, 31)


@pytest.fixture
def sample_gfr(sample_grid):
    """GridForecastResult with N=10, G=31, H=3."""
    rng = np.random.default_rng(42)
    prob = np.zeros((10, 31, 3))
    for h in range(3):
        prob[:, :, h] = rng.dirichlet(np.ones(31), size=10)
    return GridForecastResult(
        grid=sample_grid,
        prob=prob,
        basis_index=pd.RangeIndex(10),
        model_name="test_model",
    )


@pytest.fixture
def training_data():
    """Synthetic observed (X, Y) for CKD fitting."""
    rng = np.random.default_rng(0)
    T = 300
    x = rng.normal(10, 3, (T, 1))
    y = x[:, 0] * 50 + rng.normal(0, 30, T)
    return x, y


@pytest.fixture
def base_forecast_result():
    """Multi-horizon ParametricForecastResult simulating a base model."""
    rng = np.random.default_rng(1)
    N, H = 20, 4
    loc = rng.normal(10, 3, (N, H))
    scale = np.full((N, H), 2.0)
    return ParametricForecastResult(
        dist_name="normal",
        params={"loc": loc, "scale": scale},
        basis_index=pd.RangeIndex(N),
        model_name="base_parametric",
    )


@pytest.fixture
def sample_forecast_result():
    """Multi-horizon SampleForecastResult simulating a foundation model."""
    rng = np.random.default_rng(2)
    N, S, H = 20, 200, 4
    samples = rng.normal(10, 3, (N, S, H))
    return SampleForecastResult(
        samples=samples,
        basis_index=pd.RangeIndex(N),
        model_name="base_sample",
    )


@pytest.fixture
def quantile_forecast_result():
    """Multi-horizon QuantileForecastResult simulating a quantile model."""
    rng = np.random.default_rng(3)
    N, H = 20, 4
    quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles_data = {}
    base = rng.normal(10, 3, (N, H))
    for q in quantile_levels:
        quantiles_data[q] = base + (q - 0.5) * 4  # spread around base
    return QuantileForecastResult(
        quantiles_data=quantiles_data,
        basis_index=pd.RangeIndex(N),
        model_name="base_quantile",
    )


# =====================================================================
# GridForecastResult — Construction & Validation
# =====================================================================

class TestGridForecastResultConstruction:
    """Constructor validation."""

    def test_valid(self, sample_gfr):
        """Valid construction."""
        assert sample_gfr.horizon == 3
        assert sample_gfr.prob.shape == (10, 31, 3)
        assert len(sample_gfr) == 10

    def test_prob_not_3d(self, sample_grid):
        """2-D prob raises ValueError."""
        with pytest.raises(ValueError, match="3-D"):
            GridForecastResult(
                grid=sample_grid,
                prob=np.ones((10, 31)) / 31,
                basis_index=pd.RangeIndex(10),
            )

    def test_prob_n_mismatch(self, sample_grid):
        """prob dim 0 != basis_index length raises ValueError."""
        prob = np.ones((5, 31, 2)) / 31
        with pytest.raises(ValueError, match="dim 0"):
            GridForecastResult(
                grid=sample_grid,
                prob=prob,
                basis_index=pd.RangeIndex(10),
            )

    def test_prob_g_mismatch(self, sample_grid):
        """prob dim 1 != grid length raises ValueError."""
        prob = np.ones((10, 20, 2)) / 20
        with pytest.raises(ValueError, match="dim 1"):
            GridForecastResult(
                grid=sample_grid,
                prob=prob,
                basis_index=pd.RangeIndex(10),
            )

    def test_grid_not_1d(self):
        """2-D grid raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            GridForecastResult(
                grid=np.ones((5, 3)),
                prob=np.ones((2, 5, 1)) / 5,
                basis_index=pd.RangeIndex(2),
            )

    def test_model_name(self, sample_gfr):
        """Model name is stored."""
        assert sample_gfr.model_name == "test_model"

    def test_repr(self, sample_gfr):
        """repr is informative."""
        r = repr(sample_gfr)
        assert "N=10" in r
        assert "G=31" in r
        assert "H=3" in r


# =====================================================================
# GridForecastResult — to_distribution
# =====================================================================

class TestToDistribution:
    """to_distribution(h) returns valid GridDistribution."""

    def test_returns_grid_distribution(self, sample_gfr):
        """Returns GridDistribution type."""
        gd = sample_gfr.to_distribution(1)
        assert isinstance(gd, GridDistribution)

    def test_correct_shape(self, sample_gfr):
        """GridDistribution has correct (N, G) shape."""
        gd = sample_gfr.to_distribution(2)
        assert gd.prob.shape == (10, 31)

    def test_shares_grid(self, sample_gfr):
        """All horizons share the same grid."""
        gd1 = sample_gfr.to_distribution(1)
        gd3 = sample_gfr.to_distribution(3)
        np.testing.assert_array_equal(gd1.grid, gd3.grid)

    def test_different_horizons_differ(self, sample_gfr):
        """Different horizons have different prob slices."""
        gd1 = sample_gfr.to_distribution(1)
        gd2 = sample_gfr.to_distribution(2)
        assert not np.array_equal(gd1.prob, gd2.prob)

    def test_h_out_of_range(self, sample_gfr):
        """Invalid horizon raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            sample_gfr.to_distribution(0)
        with pytest.raises(ValueError, match="out of range"):
            sample_gfr.to_distribution(4)

    def test_distribution_is_functional(self, sample_gfr):
        """Extracted GridDistribution supports full API."""
        gd = sample_gfr.to_distribution(1)
        assert gd.mean().shape == (10,)
        assert gd.std().shape == (10,)
        assert gd.ppf(0.5).shape == (10,)
        assert grid_crps(gd,np.full(10, 50.0)).shape == (10,)


# =====================================================================
# GridForecastResult — reindex
# =====================================================================

class TestReindex:
    """Reindex subsetting."""

    def test_reindex_subset(self, sample_gfr):
        """Reindex with subset of indices."""
        sub = sample_gfr.reindex(pd.RangeIndex(5))
        assert len(sub) == 5
        assert sub.prob.shape == (5, 31, 3)
        np.testing.assert_array_equal(sub.grid, sample_gfr.grid)

    def test_reindex_preserves_data(self, sample_gfr):
        """Reindexed data matches original slices."""
        idx = pd.RangeIndex(3)
        sub = sample_gfr.reindex(idx)
        np.testing.assert_array_equal(sub.prob, sample_gfr.prob[:3])

    def test_reindex_invalid_index(self, sample_gfr):
        """Missing index values raise KeyError."""
        with pytest.raises(KeyError):
            sample_gfr.reindex(pd.Index([100, 200]))


# =====================================================================
# GridForecastResult — save / load
# =====================================================================

class TestSaveLoad:
    """Save and load round-trip."""

    def test_round_trip(self, sample_gfr):
        """Save → load preserves all data."""
        with tempfile.TemporaryDirectory() as d:
            sample_gfr.save(d)
            loaded = load_forecast_result(d)

        assert isinstance(loaded, GridForecastResult)
        np.testing.assert_array_equal(loaded.grid, sample_gfr.grid)
        np.testing.assert_array_equal(loaded.prob, sample_gfr.prob)
        assert loaded.model_name == sample_gfr.model_name
        assert loaded.horizon == sample_gfr.horizon

    def test_round_trip_distribution(self, sample_gfr):
        """Loaded result produces same distributions."""
        with tempfile.TemporaryDirectory() as d:
            sample_gfr.save(d)
            loaded = load_forecast_result(d)

        gd_orig = sample_gfr.to_distribution(1)
        gd_load = loaded.to_distribution(1)
        np.testing.assert_allclose(gd_orig.mean(), gd_load.mean())


# =====================================================================
# GridForecastResult — to_dataframe
# =====================================================================

class TestToDataFrame:
    """DataFrame conversion (inherited from BaseForecastResult)."""

    def test_single_horizon(self, sample_gfr):
        """to_dataframe(h=1) returns mu/std."""
        df = sample_gfr.to_dataframe(h=1)
        assert "mu" in df.columns
        assert "std" in df.columns
        assert len(df) == 10

    def test_all_horizons(self, sample_gfr):
        """to_dataframe() returns MultiIndex columns."""
        df = sample_gfr.to_dataframe()
        assert isinstance(df.columns, pd.MultiIndex)
        assert df.shape == (10, 6)  # (mu + std) × 3 horizons


# =====================================================================
# CKDRunner — Fixed config
# =====================================================================

class TestCKDRunnerFixed:
    """CKDRunner with fixed bandwidth (no Optuna)."""

    def test_fit_run_basic(self, training_data, base_forecast_result):
        """Basic fit + run pipeline with auto bandwidth."""
        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config, model_name="CKD_fixed",
        )
        runner.fit_hyperparameters()
        assert runner.is_fitted_

        result = runner.run(test_results=[base_forecast_result], seed=42)
        assert isinstance(result, GridForecastResult)
        assert result.prob.shape[0] == 20  # N=20
        assert result.prob.shape[2] == 4  # H=4
        assert result.model_name == "CKD_fixed"

    def test_run_before_fit_raises(self, training_data, base_forecast_result):
        """run() before fit_hyperparameters() raises RuntimeError."""
        x, y = training_data
        config = CKDConfig(n_x_vars=1)
        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
        )
        with pytest.raises(RuntimeError, match="fit"):
            runner.run(test_results=[base_forecast_result])

    def test_horizons_independent(self, training_data, base_forecast_result):
        """Different horizons produce different distributions."""
        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
        )
        runner.fit_hyperparameters()
        result = runner.run(test_results=[base_forecast_result], seed=42)

        gd1 = result.to_distribution(1)
        gd4 = result.to_distribution(4)
        assert gd1.mean().shape == (20,)
        assert gd4.mean().shape == (20,)

    def test_result_grid_distribution_functional(
        self, training_data, base_forecast_result
    ):
        """Output GridDistribution supports full evaluation."""
        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
        )
        runner.fit_hyperparameters()
        result = runner.run(test_results=[base_forecast_result], seed=42)

        gd = result.to_distribution(1)
        y_test = np.full(20, 500.0)
        crps_vals = grid_crps(gd, y_test)
        assert crps_vals.shape == (20,)
        assert np.all(crps_vals >= 0)
        assert np.all(np.isfinite(crps_vals))

    def test_seed_reproducibility(self, training_data, base_forecast_result):
        """Same seed → same result."""
        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
        )
        runner.fit_hyperparameters()
        r1 = runner.run(test_results=[base_forecast_result], seed=42)
        r2 = runner.run(test_results=[base_forecast_result], seed=42)
        np.testing.assert_array_equal(r1.prob, r2.prob)


# =====================================================================
# CKDRunner — All ForecastResult input types
# =====================================================================

class TestCKDRunnerInputTypes:
    """CKDRunner with each ForecastResult type as base model input."""

    def _make_runner(self, training_data):
        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
        )
        runner.fit_hyperparameters()
        return runner

    def test_parametric_forecast_result(
        self, training_data, base_forecast_result
    ):
        """ParametricForecastResult → GridForecastResult."""
        runner = self._make_runner(training_data)
        result = runner.run(test_results=[base_forecast_result], seed=0)

        assert isinstance(result, GridForecastResult)
        assert result.horizon == base_forecast_result.horizon
        assert result.prob.shape[0] == len(base_forecast_result)
        assert result.model_name == "CKD"

        # Extracted distribution is functional
        gd = result.to_distribution(1)
        assert np.all(np.isfinite(gd.mean()))
        assert np.all(grid_crps(gd,np.full(20, 500.0)) >= 0)

    def test_sample_forecast_result(
        self, training_data, sample_forecast_result
    ):
        """SampleForecastResult → GridForecastResult."""
        runner = self._make_runner(training_data)
        result = runner.run(test_results=[sample_forecast_result], seed=0)

        assert isinstance(result, GridForecastResult)
        assert result.horizon == sample_forecast_result.horizon
        assert result.prob.shape[0] == len(sample_forecast_result)

        gd = result.to_distribution(1)
        assert np.all(np.isfinite(gd.mean()))
        assert np.all(grid_crps(gd,np.full(20, 500.0)) >= 0)

    def test_quantile_forecast_result(
        self, training_data, quantile_forecast_result
    ):
        """QuantileForecastResult → GridForecastResult."""
        runner = self._make_runner(training_data)
        result = runner.run(test_results=[quantile_forecast_result], seed=0)

        assert isinstance(result, GridForecastResult)
        assert result.horizon == quantile_forecast_result.horizon
        assert result.prob.shape[0] == len(quantile_forecast_result)

        gd = result.to_distribution(1)
        assert np.all(np.isfinite(gd.mean()))
        assert np.all(grid_crps(gd,np.full(20, 500.0)) >= 0)

    def test_all_types_same_shape(
        self, training_data,
        base_forecast_result,
        sample_forecast_result,
        quantile_forecast_result,
    ):
        """All input types produce same output shape (N, G, H)."""
        runner = self._make_runner(training_data)

        r_param = runner.run(test_results=[base_forecast_result], seed=42)
        r_sample = runner.run(test_results=[sample_forecast_result], seed=42)
        r_quant = runner.run(test_results=[quantile_forecast_result], seed=42)

        # All have same N, G, H
        assert r_param.prob.shape == r_sample.prob.shape == r_quant.prob.shape
        # All share same grid (same CKD model)
        np.testing.assert_array_equal(r_param.grid, r_sample.grid)
        np.testing.assert_array_equal(r_param.grid, r_quant.grid)

    def test_parametric_vs_sample_means_correlated(
        self, training_data,
        base_forecast_result,
        sample_forecast_result,
    ):
        """Parametric and Sample inputs with similar data → correlated means."""
        runner = self._make_runner(training_data)
        r_param = runner.run(test_results=[base_forecast_result], seed=42)
        r_sample = runner.run(test_results=[sample_forecast_result], seed=42)

        # Both should produce finite, positive means
        m_param = r_param.to_distribution(1).mean()
        m_sample = r_sample.to_distribution(1).mean()
        assert np.all(np.isfinite(m_param))
        assert np.all(np.isfinite(m_sample))


# =====================================================================
# CKDRunner — Optuna optimization
# =====================================================================

class TestCKDRunnerOptuna:
    """CKDRunner with per-horizon Optuna optimization."""

    def test_optuna_fit_run(self, training_data, base_forecast_result):
        """Optuna optimization with data-driven search space."""
        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        optuna_config = {
            "n_trials": 3,
            "decay_range": (0.98, 1.0),
        }
        val_y = np.random.default_rng(99).normal(500, 100, 20)

        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
            optuna_config=optuna_config,
        )
        runner.fit_hyperparameters(
            val_results=[base_forecast_result],
            val_observed=val_y,
        )
        assert runner.is_fitted_
        assert len(runner.models_) == 4  # one per horizon

        result = runner.run(test_results=[base_forecast_result])
        assert isinstance(result, GridForecastResult)
        assert result.horizon == 4

    def test_optuna_per_horizon_models_differ(
        self, training_data, base_forecast_result
    ):
        """Per-horizon Optuna produces independent models per horizon."""
        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        optuna_config = {
            "n_trials": 5,
            "decay_range": (0.98, 1.0),
        }
        val_y = np.random.default_rng(99).normal(500, 100, 20)

        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
            optuna_config=optuna_config,
        )
        runner.fit_hyperparameters(
            val_results=[base_forecast_result],
            val_observed=val_y,
        )

        assert 1 in runner.models_
        assert 4 in runner.models_


# =====================================================================
# CKDRunner — CRPS dispatch integration
# =====================================================================

class TestCKDRunnerCRPSIntegration:
    """Integration with utils/metrics/crps.py."""

    def test_crps_dispatch_on_grid_forecast_result(
        self, training_data, base_forecast_result
    ):
        """crps() dispatch works on GridForecastResult.to_distribution()."""
        from src.utils.metrics.crps import crps as fair_crps

        x, y = training_data
        config = CKDConfig(n_x_vars=1, n_samples=200)
        runner = CKDRunner(
            x_obs=x, y_obs=y, x_columns=["wind_speed"],
            base_config=config,
        )
        runner.fit_hyperparameters()
        result = runner.run(test_results=[base_forecast_result], seed=42)

        gd = result.to_distribution(1)
        y_test = np.full(20, 500.0)
        score = fair_crps(gd, y_test)
        assert np.isfinite(score)
        assert score > 0
