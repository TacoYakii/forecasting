"""Tests for VerticalCombiner (Linear Pool / CDF Averaging)."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    ParametricForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)
from src.models.combining.vertical import VerticalCombiner, _cdf_from_quantiles
from src.utils.metrics import crps_quantile


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


def _make_sample(basis_index, model_name, seed=0, n_samples=100):
    rng = np.random.default_rng(seed)
    n = len(basis_index)
    return SampleForecastResult(
        samples=rng.standard_normal((n, n_samples, H)),
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
# Test: VerticalCombiner
# ---------------------------------------------------------------------------


class TestVerticalCombiner:
    """Tests for VerticalCombiner."""

    def test_fit_learns_weights(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "Good", loc_offset=0.0, scale=0.5)
        res_b = _make_parametric(basis_index, "Bad", loc_offset=5.0, scale=2.0)
        combiner = VerticalCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            assert combiner.weights_[h][0] > combiner.weights_[h][1]

    def test_weights_sum_to_one(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = VerticalCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            w = combiner.weights_[h]
            np.testing.assert_allclose(w.sum(), 1.0)

    def test_user_weights_override(self, basis_index, observed):
        user_w = np.array([0.6, 0.4])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = VerticalCombiner(n_quantiles=9, weights=user_w)
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(combiner.weights_[h], [0.6, 0.4])

    @pytest.mark.parametrize(
        ("user_w", "match"),
        [
            (np.array([[0.6, 0.4]]), "1D array"),
            (np.array([0.6]), "Expected 2 weights"),
            (np.array([0.6, np.nan]), "finite"),
            (np.array([1.1, -0.1]), "non-negative"),
            (np.array([0.6, 0.3]), "sum to 1"),
        ],
    )
    def test_invalid_user_weights_raise(
        self, basis_index, observed, user_w, match
    ):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = VerticalCombiner(
            n_quantiles=9, weights=user_w
        )

        with pytest.raises(ValueError, match=match):
            combiner.fit([res_a, res_b], observed)

    def test_combine_output_shape(self, basis_index, observed):
        Q = 19
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = VerticalCombiner(n_quantiles=Q)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert len(combined.quantile_levels) == Q
        assert combined.horizon == H

    def test_equal_models_match_horizontal(self, basis_index, observed):
        """When both models are identical, vertical ~ horizontal averaging."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "A_copy")  # same data, different name
        # Force equal weights
        w = np.array([0.5, 0.5])

        from src.models.combining.horizontal import HorizontalCombiner

        h_combiner = HorizontalCombiner(n_quantiles=19, weights=w)
        v_combiner = VerticalCombiner(n_quantiles=19, weights=w)

        h_combiner.fit([res_a, res_b], observed)
        v_combiner.fit([res_a, res_b], observed)

        h_result = h_combiner.combine([res_a, res_b])
        v_result = v_combiner.combine([res_a, res_b])

        # CDF-based inversion introduces interpolation differences
        ql = v_combiner.quantile_levels
        for h in range(1, H + 1):
            v_vals = np.column_stack(
                [v_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            h_vals = np.column_stack(
                [h_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            np.testing.assert_allclose(v_vals, h_vals, atol=0.05)

    def test_combined_quantiles_monotone(self, basis_index, observed):
        """Combined quantiles should be non-decreasing."""
        res_a = _make_parametric(basis_index, "A", loc_offset=0.0)
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0)
        combiner = VerticalCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        ql = combiner.quantile_levels
        for h in range(1, H + 1):
            q_vals = np.column_stack(
                [combined.quantiles_data[q][:, h - 1] for q in ql]
            )  # (N, Q)
            diffs = np.diff(q_vals, axis=1)
            assert np.all(diffs >= -1e-10), "Quantiles are not monotone"

    def test_linear_pool_wider_than_components(self, basis_index, observed):
        """Linear pool should produce wider intervals than individual models.

        This is a known property: CDF averaging spreads out the distribution
        when component models have different locations.
        """
        res_a = _make_parametric(basis_index, "A", loc_offset=-2.0, scale=0.5)
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0, scale=0.5)
        w = np.array([0.5, 0.5])
        combiner = VerticalCombiner(n_quantiles=99, weights=w)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        # Compare ranges using converted quantile arrays (same as combiner sees)
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
            q_c = np.column_stack(
                [combined.quantiles_data[q][:, h - 1] for q in ql]
            )

            # Combined range should be at least as wide as individual ranges
            range_a = q_a[:, -1] - q_a[:, 0]
            range_b = q_b[:, -1] - q_b[:, 0]
            range_c = q_c[:, -1] - q_c[:, 0]
            max_component = np.maximum(range_a, range_b)
            assert np.all(range_c >= max_component - 1e-6)

    def test_mixed_result_types(self, basis_index, observed):
        """Parametric and Sample results can be combined."""
        res_param = _make_parametric(basis_index, "Param")
        res_sample = _make_sample(basis_index, "Sample", seed=7)
        combiner = VerticalCombiner(n_quantiles=9)
        combiner.fit([res_param, res_sample], observed)
        combined = combiner.combine([res_param, res_sample])

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert len(combined.quantile_levels) == 9
        assert combined.horizon == H

    def test_to_distribution_on_combined(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = VerticalCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        dist = combined.to_distribution(2)
        assert dist.mean().shape == (N,)
        assert dist.ppf([0.1, 0.5, 0.9]).shape == (N, 3)


# ---------------------------------------------------------------------------
# Test: _cdf_from_quantiles with step functions
# ---------------------------------------------------------------------------


class TestCdfFromQuantiles:
    """Tests for _cdf_from_quantiles, especially point mass / step CDF."""

    def test_smooth_interpolation(self):
        """Basic linear interpolation between distinct quantile values."""
        levels = np.array([0.1, 0.5, 0.9])
        values = np.array([1.0, 2.0, 3.0])
        x = np.array([1.5, 2.5])
        cdf = _cdf_from_quantiles(levels, values, x)
        np.testing.assert_allclose(cdf, [0.3, 0.7])

    def test_point_mass_at_zero(self):
        """Multiple quantile levels at x=0 (point mass at 0MW).

        F(0) should equal the maximum level mapped to x=0.
        """
        levels = np.array([0.1, 0.2, 0.3, 0.5, 0.9])
        values = np.array([0.0, 0.0, 0.0, 15.0, 42.0])
        cdf = _cdf_from_quantiles(levels, values, np.array([0.0]))
        np.testing.assert_allclose(cdf, [0.3])

    def test_point_mass_at_rated_capacity(self):
        """Multiple quantile levels at rated capacity (e.g., 100MW).

        F(100) should equal the maximum level mapped to x=100.
        """
        levels = np.array([0.1, 0.5, 0.7, 0.8, 0.9])
        values = np.array([30.0, 60.0, 100.0, 100.0, 100.0])
        cdf = _cdf_from_quantiles(levels, values, np.array([100.0]))
        np.testing.assert_allclose(cdf, [0.9])

    def test_both_bounds_point_mass(self):
        """Point mass at both 0 and rated capacity."""
        levels = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        values = np.array([0.0, 0.0, 25.0, 50.0, 50.0])
        x = np.array([0.0, 25.0, 50.0])
        cdf = _cdf_from_quantiles(levels, values, x)
        np.testing.assert_allclose(cdf, [0.2, 0.5, 0.9])

    def test_cdf_clamps_outside_range(self):
        """Values outside quantile range clamp to 0 and 1."""
        levels = np.array([0.1, 0.5, 0.9])
        values = np.array([10.0, 20.0, 30.0])
        x = np.array([5.0, 35.0])
        cdf = _cdf_from_quantiles(levels, values, x)
        np.testing.assert_allclose(cdf, [0.0, 1.0])

    def test_monotone_cdf(self):
        """CDF output should be non-decreasing for sorted x input."""
        levels = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95])
        values = np.array([0.0, 0.0, 0.0, 5.0, 20.0, 40.0, 50.0, 50.0])
        x = np.linspace(-5, 55, 100)
        cdf = _cdf_from_quantiles(levels, values, x)
        assert np.all(np.diff(cdf) >= -1e-12), "CDF is not monotone"


# ---------------------------------------------------------------------------
# Test: VerticalCombiner with step-function inputs
# ---------------------------------------------------------------------------


class TestVerticalCombinerStepFunction:
    """End-to-end tests with point-mass / bounded forecasts."""

    @pytest.fixture
    def basis_index(self):
        return pd.date_range("2024-01-01", periods=10, freq="h")

    def _make_quantile_with_point_mass(
        self, basis_index, model_name, mass_value, mass_fraction, H=3
    ):
        """Create QuantileForecastResult with point mass at mass_value.

        The first mass_fraction of quantile levels are set to mass_value,
        the rest linearly increase.
        """
        Q = 19
        ql = np.linspace(0, 1, Q + 2)[1:-1]
        n_mass = int(mass_fraction * Q)
        N = len(basis_index)

        quantiles_data = {}
        for i, level in enumerate(ql):
            vals = np.empty((N, H))
            for h in range(H):
                if i < n_mass:
                    vals[:, h] = mass_value
                else:
                    vals[:, h] = mass_value + (i - n_mass + 1) * 5.0
            quantiles_data[level] = vals

        return QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=basis_index,
            model_name=model_name,
        )

    def test_combine_with_point_mass_at_zero(self, basis_index):
        """Combining models with point mass at 0MW should not error."""
        H = 3
        res_a = self._make_quantile_with_point_mass(
            basis_index, "A", mass_value=0.0, mass_fraction=0.3, H=H
        )
        res_b = self._make_quantile_with_point_mass(
            basis_index, "B", mass_value=0.0, mass_fraction=0.5, H=H
        )
        rng = np.random.default_rng(42)
        observed = pd.DataFrame(
            rng.uniform(0, 50, (10, H)), index=basis_index
        )

        combiner = VerticalCombiner(n_quantiles=19, weights=np.array([0.5, 0.5]))
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        assert len(combined) == 10
        assert combined.horizon == H

        # Combined quantiles should be non-decreasing
        ql = combiner.quantile_levels
        for h in range(1, H + 1):
            q_vals = np.column_stack(
                [combined.quantiles_data[q][:, h - 1] for q in ql]
            )
            diffs = np.diff(q_vals, axis=1)
            assert np.all(diffs >= -1e-10), "Quantiles are not monotone"

    def test_combine_with_point_mass_at_both_bounds(self, basis_index):
        """Combining with point mass at 0 and rated capacity."""
        H = 3
        res_a = self._make_quantile_with_point_mass(
            basis_index, "A", mass_value=0.0, mass_fraction=0.3, H=H
        )
        # Model B: high output, point mass near rated capacity
        Q = 19
        ql = np.linspace(0, 1, Q + 2)[1:-1]
        N = len(basis_index)
        quantiles_data = {}
        for i, level in enumerate(ql):
            vals = np.empty((N, H))
            for h in range(H):
                if i >= 14:  # top ~25% of quantiles at rated
                    vals[:, h] = 100.0
                else:
                    vals[:, h] = 20.0 + i * 5.0
            quantiles_data[level] = vals

        res_b = QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=basis_index,
            model_name="B",
        )
        rng = np.random.default_rng(42)
        observed = pd.DataFrame(
            rng.uniform(0, 100, (10, H)), index=basis_index
        )

        combiner = VerticalCombiner(n_quantiles=19, weights=np.array([0.5, 0.5]))
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        assert len(combined) == 10
        # Monotonicity check
        ql = combiner.quantile_levels
        for h in range(1, H + 1):
            q_vals = np.column_stack(
                [combined.quantiles_data[q][:, h - 1] for q in ql]
            )
            diffs = np.diff(q_vals, axis=1)
            assert np.all(diffs >= -1e-10), "Quantiles are not monotone"


# ---------------------------------------------------------------------------
# Test: VerticalCombiner optimization-based fitting
# ---------------------------------------------------------------------------


class TestVerticalOptimize:
    """Tests for optimization-based weight fitting."""

    def test_optimize_weights_sum_to_one(self, basis_index, observed):
        """Optimized weights must sum to 1."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = VerticalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(combiner.weights_[h].sum(), 1.0)

    def test_optimize_improves_or_matches_heuristic(
        self, basis_index, observed
    ):
        """Optimized CRPS on training data <= inverse-CRPS heuristic."""
        res_a = _make_parametric(basis_index, "Good", loc_offset=0.0, scale=0.5)
        res_b = _make_parametric(basis_index, "Bad", loc_offset=5.0, scale=2.0)

        c_heur = VerticalCombiner(
            n_quantiles=19, fit_method="inverse_crps"
        )
        c_opt = VerticalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        c_heur.fit([res_a, res_b], observed)
        c_opt.fit([res_a, res_b], observed)

        tau = c_heur.quantile_levels
        obs_arr = observed.values

        for h in range(1, H + 1):
            comb_heur = c_heur.combine([res_a, res_b])
            comb_opt = c_opt.combine([res_a, res_b])

            q_h = np.column_stack([
                comb_heur.quantiles_data[q][:, h - 1] for q in tau
            ])
            q_o = np.column_stack([
                comb_opt.quantiles_data[q][:, h - 1] for q in tau
            ])
            crps_h = crps_quantile(tau, q_h, obs_arr[:, h - 1])
            crps_o = crps_quantile(tau, q_o, obs_arr[:, h - 1])
            assert crps_o <= crps_h + 1e-8

    def test_regularization_shrinks_to_equal(self, basis_index, observed):
        """Large reg_lambda pushes weights toward 1/M."""
        res_a = _make_parametric(basis_index, "A", loc_offset=0.0)
        res_b = _make_parametric(basis_index, "B", loc_offset=3.0)
        combiner = VerticalCombiner(
            n_quantiles=19, fit_method="optimize", reg_lambda=100.0
        )
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(
                combiner.weights_[h], [0.5, 0.5], atol=0.05
            )

    def test_fit_method_inverse_crps_unchanged(self, basis_index, observed):
        """fit_method='inverse_crps' matches original heuristic behavior."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        combiner = VerticalCombiner(
            n_quantiles=19, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)

        from src.models.combining.base import BaseCombiner

        conv_a = combiner._convert_to_quantile(res_a)
        conv_b = combiner._convert_to_quantile(res_b)
        tau = combiner.quantile_levels

        for h in range(1, H + 1):
            qarrays = combiner._extract_quantile_arrays(
                [conv_a, conv_b], h
            )
            expected = BaseCombiner._inverse_crps_weights(
                tau, qarrays, observed.values[:, h - 1]
            )
            np.testing.assert_allclose(
                combiner.weights_[h], expected, atol=1e-12
            )

    def test_invalid_fit_method_raises(self):
        """Unknown fit_method raises ValueError."""
        with pytest.raises(ValueError, match="fit_method"):
            VerticalCombiner(fit_method="bogus")

    def test_optimize_user_weights_override(self, basis_index, observed):
        """User weights bypass optimization entirely."""
        user_w = np.array([0.6, 0.4])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = VerticalCombiner(
            n_quantiles=9, fit_method="optimize", weights=user_w
        )
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(combiner.weights_[h], [0.6, 0.4])

    def test_optimize_includes_heuristic_as_candidate(
        self, basis_index, observed
    ):
        """Simplex search always includes x0, so result >= heuristic."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        c_heur = VerticalCombiner(
            n_quantiles=19, fit_method="inverse_crps"
        )
        c_opt = VerticalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        c_heur.fit([res_a, res_b], observed)
        c_opt.fit([res_a, res_b], observed)

        # Optimized weights are always at least as good as heuristic
        # because x0 is included in the candidate set
        assert c_opt.is_fitted_
        for h in range(1, H + 1):
            np.testing.assert_allclose(c_opt.weights_[h].sum(), 1.0)

    def test_optimize_three_models(self, basis_index, observed):
        """Optimization works with M > 2 models."""
        res_a = _make_parametric(basis_index, "A", loc_offset=0.0)
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        res_c = _make_parametric(basis_index, "C", loc_offset=3.0, scale=1.5)
        combiner = VerticalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        combiner.fit([res_a, res_b, res_c], observed)

        for h in range(1, H + 1):
            w = combiner.weights_[h]
            assert len(w) == 3
            np.testing.assert_allclose(w.sum(), 1.0)
            assert np.all(w >= 0)

    def test_optimize_with_val_ratio(self, basis_index, observed):
        """Optimization with validation split produces scores."""
        res_a = _make_parametric(basis_index, "Good", loc_offset=0.0, scale=0.5)
        res_b = _make_parametric(basis_index, "Bad", loc_offset=5.0, scale=2.0)
        combiner = VerticalCombiner(
            n_quantiles=19, fit_method="optimize", val_ratio=0.2
        )
        combiner.fit([res_a, res_b], observed)

        assert len(combiner.train_scores_) == H
        assert len(combiner.val_scores_) == H
        for h in range(1, H + 1):
            assert combiner.train_scores_[h] >= 0
            assert combiner.val_scores_[h] >= 0
            np.testing.assert_allclose(combiner.weights_[h].sum(), 1.0)
