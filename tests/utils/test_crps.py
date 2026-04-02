"""Tests for CRPS metrics: pinball_loss, crps_quantile, and crps dispatcher."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.core.forecast_results import (
    ParametricForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)
from src.utils.metrics import crps, crps_quantile, pinball_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, H = 50, 3
Q = 99


@pytest.fixture
def basis_index():
    return pd.date_range("2024-01-01", periods=N, freq="h")


@pytest.fixture
def observed(basis_index):
    rng = np.random.default_rng(42)
    return rng.standard_normal(N)


@pytest.fixture
def tau():
    return np.linspace(0, 1, Q + 2)[1:-1]


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------


class TestPinballLoss:
    """Tests for pinball_loss reduction modes."""

    def test_reduction_none_shape(self, tau):
        q = np.sort(np.random.default_rng(0).standard_normal((10, Q)), axis=1)
        y = np.zeros(10)
        result = pinball_loss(tau, q, y, reduction="none")
        assert result.shape == (10, Q)

    def test_reduction_obs_shape(self, tau):
        q = np.sort(np.random.default_rng(0).standard_normal((10, Q)), axis=1)
        y = np.zeros(10)
        result = pinball_loss(tau, q, y, reduction="obs")
        assert result.shape == (10,)

    def test_reduction_mean_scalar(self, tau):
        q = np.sort(np.random.default_rng(0).standard_normal((10, Q)), axis=1)
        y = np.zeros(10)
        result = pinball_loss(tau, q, y, reduction="mean")
        assert isinstance(result, float)

    def test_invalid_reduction_raises(self, tau):
        with pytest.raises(ValueError, match="Unknown reduction"):
            pinball_loss(tau, np.zeros((1, Q)), np.zeros(1), reduction="bad")


# ---------------------------------------------------------------------------
# crps_quantile
# ---------------------------------------------------------------------------


class TestCRPSQuantile:
    """Tests for crps_quantile."""

    def test_perfect_forecast_zero(self, tau):
        """CRPS ~0 when all quantiles equal observation."""
        y = np.array([5.0])
        q = np.full((1, Q), 5.0)
        score = crps_quantile(tau, q, y, reduction="mean")
        np.testing.assert_allclose(score, 0.0, atol=1e-10)

    def test_nonuniform_tau(self):
        """Non-uniform tau grid should still produce valid CRPS."""
        tau_nonuniform = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        y = np.array([0.0])
        q = norm(0, 1).ppf(tau_nonuniform)[None, :]
        score = crps_quantile(tau_nonuniform, q, y, reduction="mean")
        assert score > 0
        assert np.isfinite(score)

    def test_invalid_reduction_raises(self, tau):
        with pytest.raises(ValueError, match="reduction must be"):
            crps_quantile(tau, np.zeros((1, Q)), np.zeros(1), reduction="bad")

    def test_unsorted_tau_raises(self):
        """Non-monotone tau should raise ValueError."""
        tau_bad = np.array([0.9, 0.1, 0.5])
        with pytest.raises(ValueError, match="monotonically increasing"):
            crps_quantile(tau_bad, np.zeros((1, 3)), np.zeros(1))

    def test_single_tau_raises(self):
        """Single-element tau should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            crps_quantile(np.array([0.5]), np.zeros((1, 1)), np.zeros(1))

    def test_tau_out_of_range_raises(self):
        """tau values outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="open interval"):
            crps_quantile(np.array([-0.1, 0.5, 0.9]), np.zeros((1, 3)), np.zeros(1))
        with pytest.raises(ValueError, match="open interval"):
            crps_quantile(np.array([0.1, 0.5, 1.0]), np.zeros((1, 3)), np.zeros(1))

    def test_shape_mismatch_raises(self):
        """Mismatched q/y/tau shapes should raise ValueError."""
        tau = np.array([0.1, 0.5, 0.9])
        # q has wrong number of quantile columns
        with pytest.raises(ValueError, match="q must have shape"):
            crps_quantile(tau, np.zeros((5, 2)), np.zeros(5))
        # q has wrong number of observations
        with pytest.raises(ValueError, match="q must have shape"):
            crps_quantile(tau, np.zeros((3, 3)), np.zeros(5))
        # y is 2-D
        with pytest.raises(ValueError, match="y must be 1-D"):
            crps_quantile(tau, np.zeros((5, 3)), np.zeros((5, 1)))


# ---------------------------------------------------------------------------
# crps dispatcher
# ---------------------------------------------------------------------------


class TestCRPSDispatch:
    """Tests for crps() dispatch function across ForecastResult types."""

    def test_parametric_result(self, basis_index, observed):
        """Dispatch on ParametricForecastResult uses ppf -> crps_numerical."""
        rng = np.random.default_rng(0)
        result = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": rng.standard_normal((N, H)),
                "scale": np.full((N, H), 1.0),
            },
            basis_index=basis_index,
        )
        score = crps(result, observed, h=1)
        assert isinstance(score, float)
        assert score > 0
        assert np.isfinite(score)

    def test_sample_result(self, basis_index, observed):
        """Dispatch on SampleForecastResult uses percentile -> crps_numerical."""
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((N, 200, H))
        result = SampleForecastResult(
            samples=samples,
            basis_index=basis_index,
        )
        score = crps(result, observed, h=1)
        assert isinstance(score, float)
        assert score > 0
        assert np.isfinite(score)

    def test_quantile_result(self, basis_index, observed):
        """Dispatch on QuantileForecastResult uses ppf -> crps_numerical."""
        tau = np.linspace(0, 1, Q + 2)[1:-1]
        rng = np.random.default_rng(0)
        quantiles_data = {}
        for level in tau:
            quantiles_data[float(level)] = (
                rng.standard_normal((N, H)) + norm.ppf(level)
            )
        result = QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=basis_index,
        )
        score = crps(result, observed, h=1)
        assert isinstance(score, float)
        assert score > 0
        assert np.isfinite(score)

    def test_unsupported_type_raises(self, observed):
        """Non-ForecastResult raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported"):
            crps("not_a_result", observed, h=1)

    def test_invalid_horizon_raises(self, basis_index, observed):
        """Out-of-range horizon raises ValueError for all result types."""
        rng = np.random.default_rng(0)
        # SampleForecastResult (previously bypassed validation)
        sample_result = SampleForecastResult(
            samples=rng.standard_normal((N, 100, H)),
            basis_index=basis_index,
        )
        with pytest.raises(ValueError):
            crps(sample_result, observed, h=0)
        with pytest.raises(ValueError):
            crps(sample_result, observed, h=H + 1)

    def test_all_horizons(self, basis_index, observed):
        """CRPS works for all valid horizons."""
        rng = np.random.default_rng(0)
        result = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": rng.standard_normal((N, H)),
                "scale": np.full((N, H), 1.0),
            },
            basis_index=basis_index,
        )
        for h in range(1, H + 1):
            score = crps(result, observed, h=h)
            assert np.isfinite(score)

    def test_fair_comparison(self, basis_index, observed):
        """Same distribution in different result types gives similar CRPS."""
        rng = np.random.default_rng(42)
        mu = rng.standard_normal((N, H))
        sigma = np.full((N, H), 1.0)

        # Parametric
        param_result = ParametricForecastResult(
            dist_name="normal",
            params={"loc": mu, "scale": sigma},
            basis_index=basis_index,
        )

        # Sample (many samples from same distribution)
        samples = np.stack(
            [rng.normal(mu[:, hh, None], sigma[:, hh, None], size=(N, 5000))
             for hh in range(H)],
            axis=2,
        )  # (N, 5000, H)
        sample_result = SampleForecastResult(
            samples=samples, basis_index=basis_index,
        )

        # Quantile (from same distribution)
        tau = np.linspace(0, 1, 101)[1:-1]
        quantiles_data = {}
        for level in tau:
            quantiles_data[float(level)] = norm.ppf(level, loc=mu, scale=sigma)
        quantile_result = QuantileForecastResult(
            quantiles_data=quantiles_data, basis_index=basis_index,
        )

        crps_param = crps(param_result, observed, h=1)
        crps_sample = crps(sample_result, observed, h=1)
        crps_quantile_val = crps(quantile_result, observed, h=1)

        # All three should be close (same underlying distribution)
        np.testing.assert_allclose(crps_param, crps_sample, rtol=0.1)
        np.testing.assert_allclose(crps_param, crps_quantile_val, rtol=0.1)
