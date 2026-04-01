"""Tests for HorizontalCombiner (Quantile Averaging)."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    ParametricForecastResult,
    QuantileForecastResult,
)
from src.models.combining.horizontal import (
    HorizontalCombiner,
    _crps_from_quantiles,
    _pinball_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, H = 30, 3


@pytest.fixture
def basis_index():
    return pd.date_range("2024-01-01", periods=N, freq="h")


def _make_parametric(basis_index, model_name, loc_offset=0.0, scale=0.5):
    rng = np.random.default_rng(42 + int(loc_offset * 10))
    n = len(basis_index)
    return ParametricForecastResult(
        dist_name="normal",
        params={
            "loc": rng.standard_normal((n, H)) + loc_offset,
            "scale": np.full((n, H), scale),
        },
        basis_index=basis_index,
        model_name=model_name,
    )


@pytest.fixture
def observed(basis_index):
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.standard_normal((N, H)),
        index=basis_index,
    )


# ---------------------------------------------------------------------------
# Test: pinball loss and CRPS helpers
# ---------------------------------------------------------------------------


class TestPinballLoss:
    """Tests for the pinball loss helper."""

    def test_median_pinball(self):
        """Pinball loss at tau=0.5 equals 0.5 * |y - q|."""
        tau = np.array([0.5])
        q = np.array([[2.0]])
        y = np.array([3.0])
        loss = _pinball_loss(tau, q, y)
        np.testing.assert_allclose(loss, [0.5])

    def test_perfect_forecast(self):
        """Zero loss when forecast equals observation."""
        tau = np.array([0.1, 0.5, 0.9])
        y = np.array([1.0])
        q = np.array([[1.0, 1.0, 1.0]])
        loss = _pinball_loss(tau, q, y)
        np.testing.assert_allclose(loss, [0.0])


class TestCRPSFromQuantiles:
    """Tests for quantile-based CRPS approximation."""

    def test_perfect_forecast_zero_crps(self):
        """CRPS ~0 when all quantiles equal the observation."""
        tau = np.linspace(0, 1, 101)[1:-1]
        y = np.array([5.0])
        q = np.full((1, 99), 5.0)
        crps = _crps_from_quantiles(tau, q, y)
        np.testing.assert_allclose(crps, 0.0, atol=1e-10)

    def test_wider_distribution_higher_crps(self):
        """Wider distribution should have higher CRPS."""
        tau = np.linspace(0, 1, 101)[1:-1]
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)

        from scipy.stats import norm
        q_narrow = norm(0, 1).ppf(tau)[None, :].repeat(100, axis=0)
        q_wide = norm(0, 3).ppf(tau)[None, :].repeat(100, axis=0)

        crps_narrow = _crps_from_quantiles(tau, q_narrow, y)
        crps_wide = _crps_from_quantiles(tau, q_wide, y)
        assert crps_narrow < crps_wide


# ---------------------------------------------------------------------------
# Test: HorizontalCombiner
# ---------------------------------------------------------------------------


class TestHorizontalCombiner:
    """Tests for HorizontalCombiner."""

    def test_fit_learns_weights(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "Good", loc_offset=0.0, scale=0.5)
        res_b = _make_parametric(basis_index, "Bad", loc_offset=5.0, scale=2.0)
        combiner = HorizontalCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)

        # "Good" model (index 0, closer to observed) should get higher weight
        for h in range(1, H + 1):
            assert combiner.weights_[h][0] > combiner.weights_[h][1]

    def test_weights_sum_to_one(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = HorizontalCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            w = combiner.weights_[h]
            np.testing.assert_allclose(w.sum(), 1.0)

    def test_user_weights_override(self, basis_index, observed):
        user_w = np.array([0.7, 0.3])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = HorizontalCombiner(n_quantiles=9, weights=user_w)
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(combiner.weights_[h], [0.7, 0.3])

    def test_combine_output_shape(self, basis_index, observed):
        Q = 19
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = HorizontalCombiner(n_quantiles=Q)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert len(combined.quantile_levels) == Q
        assert combined.horizon == H

    def test_combine_respects_weights(self, basis_index, observed):
        """Combined quantiles match weighted average of individual quantiles."""
        user_w = np.array([0.8, 0.2])
        Q = 9
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0)
        combiner = HorizontalCombiner(n_quantiles=Q, weights=user_w)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        # Compute expected from converted quantile arrays
        conv_a = combiner._convert_to_quantile(res_a)
        conv_b = combiner._convert_to_quantile(res_b)
        ql = combiner.quantile_levels
        for h in range(1, H + 1):
            q_a = np.column_stack(
                [conv_a.quantiles_data[q][:, h - 1] for q in ql]
            )
            q_b = np.column_stack(
                [conv_b.quantiles_data[q][:, h - 1] for q in ql]
            )
            expected = 0.8 * q_a + 0.2 * q_b
            actual = np.column_stack(
                [combined.quantiles_data[q][:, h - 1] for q in ql]
            )
            np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_to_distribution_on_combined(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = HorizontalCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        dist = combined.to_distribution(1)
        assert dist.mean().shape == (N,)
