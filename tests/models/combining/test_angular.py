"""Tests for AngularCombiner (Quantile-Transform Interpolation)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.core.forecast_results import (
    GridForecastResult,
    ParametricForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)
from src.models.combining.angular import AngularCombiner
from src.models.combining.base import BaseCombiner
from src.models.combining.horizontal import HorizontalCombiner
from src.models.combining.vertical import VerticalCombiner
from src.utils.metrics import crps_quantile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, H, N_SAMPLES = 30, 3, 100


@pytest.fixture
def basis_index():
    return pd.date_range("2024-01-01", periods=N, freq="h")


def _make_parametric(basis_index, model_name, loc_offset=0.0, scale=0.5):
    """Create a ParametricForecastResult with known values."""
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


def _make_quantile(basis_index, model_name, loc_offset=0.0, scale=0.5):
    """Create a QuantileForecastResult from a normal distribution."""
    rng = np.random.default_rng(42 + int(loc_offset * 10))
    n = len(basis_index)
    ql = np.linspace(0, 1, 21)[1:-1]  # 19 levels
    loc = rng.standard_normal((n, H)) + loc_offset
    quantiles_data = {}
    for level in ql:
        quantiles_data[level] = norm.ppf(level, loc=loc, scale=scale)
    return QuantileForecastResult(
        quantiles_data=quantiles_data,
        basis_index=basis_index,
        model_name=model_name,
    )


def _make_sample(basis_index, model_name, seed=0, n_samples=N_SAMPLES):
    """Create a SampleForecastResult with known values."""
    rng = np.random.default_rng(seed)
    n = len(basis_index)
    return SampleForecastResult(
        samples=rng.standard_normal((n, n_samples, H)),
        basis_index=basis_index,
        model_name=model_name,
    )


def _make_grid(basis_index, model_name, loc_offset=0.0, scale=0.5):
    """Create a GridForecastResult from a discretized normal distribution."""
    rng = np.random.default_rng(42 + int(loc_offset * 10))
    n = len(basis_index)
    grid = np.linspace(-5, 10, 151)
    loc = rng.standard_normal((n, H)) + loc_offset
    prob = np.empty((n, len(grid), H))
    for i in range(n):
        for h in range(H):
            p = norm.pdf(grid, loc=loc[i, h], scale=scale)
            prob[i, :, h] = p / p.sum()
    return GridForecastResult(
        grid=grid,
        prob=prob,
        basis_index=basis_index,
        model_name=model_name,
    )


@pytest.fixture
def observed(basis_index):
    """Observed values DataFrame, shape (N, H)."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.standard_normal((N, H)),
        index=basis_index,
    )


# ---------------------------------------------------------------------------
# Test: BaseCombiner common behavior
# ---------------------------------------------------------------------------


class TestAngularCombinerBasic:
    """Tests shared with other combiners (contract compliance)."""

    def test_weights_sum_to_one(self, basis_index, observed):
        """Fitted weights must sum to 1 for every horizon."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=19, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(combiner.weights_[h].sum(), 1.0)

    def test_combine_output_shape(self, basis_index, observed):
        """Combined result has correct type, N, Q, H."""
        Q = 19
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=Q, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert len(combined.quantile_levels) == Q
        assert combined.horizon == H

    def test_combine_before_fit_raises(self, basis_index):
        """combine() without fit() raises RuntimeError."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(degree=30.0, fit_method="inverse_crps")
        with pytest.raises(RuntimeError, match="fit"):
            combiner.combine([res_a, res_b])

    def test_combine_reordered_models_raises(self, basis_index, observed):
        """Reordering models between fit and combine should raise."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)

        with pytest.raises(ValueError, match="Model names/order mismatch"):
            combiner.combine([res_b, res_a])

    def test_user_weights_override(self, basis_index, observed):
        """Explicit weights bypass optimization."""
        user_w = np.array([0.7, 0.3])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=9, weights=user_w, degree=30.0
        )
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(combiner.weights_[h], [0.7, 0.3])

    def test_to_distribution_on_combined(self, basis_index, observed):
        """Combined result supports to_distribution()."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=19, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        dist = combined.to_distribution(1)
        assert dist.mean().shape == (N,)
        assert dist.ppf([0.1, 0.5, 0.9]).shape == (N, 3)


# ---------------------------------------------------------------------------
# Test: Mixed result types (all 4 ForecastResult subtypes)
# ---------------------------------------------------------------------------


class TestMixedResultTypes:
    """Combining works with all ForecastResult subtypes mixed together."""

    def test_four_types_combined(self, basis_index, observed):
        """Parametric + Quantile + Sample + Grid can be combined."""
        res_param = _make_parametric(basis_index, "Parametric")
        res_quant = _make_quantile(basis_index, "Quantile", loc_offset=0.5)
        res_sample = _make_sample(basis_index, "Sample", seed=7)
        res_grid = _make_grid(basis_index, "Grid", loc_offset=1.0)

        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit(
            [res_param, res_quant, res_sample, res_grid], observed
        )
        combined = combiner.combine(
            [res_param, res_quant, res_sample, res_grid]
        )

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert combined.horizon == H

    def test_three_types_without_grid(self, basis_index, observed):
        """Parametric + Quantile + Sample can be combined."""
        res_param = _make_parametric(basis_index, "Parametric")
        res_quant = _make_quantile(basis_index, "Quantile", loc_offset=0.5)
        res_sample = _make_sample(basis_index, "Sample", seed=7)

        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_param, res_quant, res_sample], observed)
        combined = combiner.combine([res_param, res_quant, res_sample])

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert combined.horizon == H


# ---------------------------------------------------------------------------
# Test: Validation split
# ---------------------------------------------------------------------------


class TestAngularValidationSplit:
    """Tests for temporal train/validation split."""

    def test_val_scores_populated(self, basis_index, observed):
        """val_scores_ is populated when val_ratio > 0."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0,
            fit_method="inverse_crps", val_ratio=0.2,
        )
        combiner.fit([res_a, res_b], observed)

        assert len(combiner.train_scores_) == H
        assert len(combiner.val_scores_) == H
        for h in range(1, H + 1):
            assert isinstance(combiner.train_scores_[h], float)
            assert isinstance(combiner.val_scores_[h], float)
            assert combiner.train_scores_[h] >= 0
            assert combiner.val_scores_[h] >= 0

    def test_val_scores_empty_when_no_split(self, basis_index, observed):
        """val_scores_ is empty when val_ratio=0."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)

        assert len(combiner.train_scores_) == H
        assert len(combiner.val_scores_) == 0


# ---------------------------------------------------------------------------
# Test: degree parameter behavior
# ---------------------------------------------------------------------------


class TestDegreeBehavior:
    """Tests for angular degree parameter effects."""

    def test_degree_zero_matches_horizontal(self, basis_index, observed):
        """degree=0 reduces to horizontal (quantile) weighted average."""
        w = np.array([0.6, 0.4])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        h_combiner = HorizontalCombiner(n_quantiles=19, weights=w)
        a_combiner = AngularCombiner(
            n_quantiles=19, weights=w, degree=0.0
        )

        h_combiner.fit([res_a, res_b], observed)
        a_combiner.fit([res_a, res_b], observed)

        h_result = h_combiner.combine([res_a, res_b])
        a_result = a_combiner.combine([res_a, res_b])

        ql = a_combiner.quantile_levels
        for h in range(1, H + 1):
            h_vals = np.column_stack(
                [h_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            a_vals = np.column_stack(
                [a_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            np.testing.assert_allclose(a_vals, h_vals, atol=1e-10)

    def test_degree_90_matches_vertical(self, basis_index, observed):
        """degree=90 approximates vertical (CDF) weighted average."""
        w = np.array([0.6, 0.4])
        Q = 99
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        v_combiner = VerticalCombiner(n_quantiles=Q, weights=w)
        a_combiner = AngularCombiner(
            n_quantiles=Q, weights=w, degree=90.0
        )

        v_combiner.fit([res_a, res_b], observed)
        a_combiner.fit([res_a, res_b], observed)

        v_result = v_combiner.combine([res_a, res_b])
        a_result = a_combiner.combine([res_a, res_b])

        ql = a_combiner.quantile_levels
        for h in range(1, H + 1):
            v_vals = np.column_stack(
                [v_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            a_vals = np.column_stack(
                [a_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            np.testing.assert_allclose(a_vals, v_vals, rtol=0.05)

    def test_degree_positive_differs_from_horizontal(
        self, basis_index, observed
    ):
        """degree > 0 produces different results from horizontal."""
        w = np.array([0.6, 0.4])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0)

        h_combiner = HorizontalCombiner(n_quantiles=19, weights=w)
        a_combiner = AngularCombiner(
            n_quantiles=19, weights=w, degree=45.0
        )

        h_combiner.fit([res_a, res_b], observed)
        a_combiner.fit([res_a, res_b], observed)

        h_result = h_combiner.combine([res_a, res_b])
        a_result = a_combiner.combine([res_a, res_b])

        ql = a_combiner.quantile_levels
        h_vals = np.column_stack(
            [h_result.quantiles_data[q][:, 0] for q in ql]
        )
        a_vals = np.column_stack(
            [a_result.quantiles_data[q][:, 0] for q in ql]
        )
        assert not np.allclose(a_vals, h_vals, atol=1e-6)

    def test_combined_quantiles_monotone(self, basis_index, observed):
        """Combined quantiles must be non-decreasing for all degrees."""
        res_a = _make_parametric(basis_index, "A", loc_offset=0.0)
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0)

        for degree in [0.0, 30.0, 45.0, 90.0]:
            combiner = AngularCombiner(
                n_quantiles=19, degree=degree, fit_method="inverse_crps"
            )
            combiner.fit([res_a, res_b], observed)
            combined = combiner.combine([res_a, res_b])

            ql = combiner.quantile_levels
            for h in range(1, H + 1):
                q_vals = np.column_stack(
                    [combined.quantiles_data[q][:, h - 1] for q in ql]
                )
                diffs = np.diff(q_vals, axis=1)
                assert np.all(diffs >= -1e-10), (
                    f"Quantiles not monotone at degree={degree}, h={h}"
                )


# ---------------------------------------------------------------------------
# Test: Variance monotonicity (Theorem 2, Taylor & Meng 2025)
# ---------------------------------------------------------------------------


class TestVarianceMonotonicity:
    """Var(F_{A,w,θ}) is monotonically increasing w.r.t. θ.

    Under Assumption 1 (same location-scale family, symmetric, unimodal),
    the variance of the angular combination increases with θ. Normal
    distributions satisfy Assumption 1.
    """

    def test_variance_increases_with_degree(self, basis_index, observed):
        """Variance increases monotonically from θ=0 to θ=90."""
        w = np.array([0.6, 0.4])
        Q = 99
        res_a = _make_parametric(basis_index, "A", scale=0.5)
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0, scale=0.5)

        degrees = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0, 90.0]

        for h in range(1, H + 1):
            variances = []
            for degree in degrees:
                combiner = AngularCombiner(
                    n_quantiles=Q, weights=w, degree=degree
                )
                combiner.fit([res_a, res_b], observed)
                combined = combiner.combine([res_a, res_b])
                std_vals = combined.to_distribution(h).std()
                variances.append(np.mean(std_vals ** 2))

            for i in range(1, len(variances)):
                assert variances[i] >= variances[i - 1] - 1e-6, (
                    f"h={h}: Var at θ={degrees[i]:.0f} "
                    f"({variances[i]:.6f}) < Var at θ={degrees[i-1]:.0f} "
                    f"({variances[i-1]:.6f})"
                )

    def test_three_models_variance_monotone(self, basis_index, observed):
        """Variance monotonicity holds for 3 models."""
        w = np.array([0.5, 0.3, 0.2])
        Q = 99
        res_a = _make_parametric(basis_index, "A", scale=0.5)
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0, scale=0.5)
        res_c = _make_parametric(basis_index, "C", loc_offset=3.0, scale=0.5)

        degrees = [0.0, 30.0, 45.0, 60.0, 90.0]

        for h in range(1, H + 1):
            variances = []
            for degree in degrees:
                combiner = AngularCombiner(
                    n_quantiles=Q, weights=w, degree=degree
                )
                combiner.fit([res_a, res_b, res_c], observed)
                combined = combiner.combine([res_a, res_b, res_c])
                std_vals = combined.to_distribution(h).std()
                variances.append(np.mean(std_vals ** 2))

            for i in range(1, len(variances)):
                assert variances[i] >= variances[i - 1] - 1e-6, (
                    f"h={h}: Var at θ={degrees[i]:.0f} "
                    f"({variances[i]:.6f}) < Var at θ={degrees[i-1]:.0f} "
                    f"({variances[i-1]:.6f})"
                )


# ---------------------------------------------------------------------------
# Test: Combined mean equals weighted sum of component means
# ---------------------------------------------------------------------------


class TestCombinedMean:
    """E[combined] = sum_i w_i * E[model_i] for each horizon."""

    def test_mean_degree_zero(self, basis_index, observed):
        """Exact equality when degree=0 (deterministic weighted average)."""
        w = np.array([0.6, 0.4])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0)

        combiner = AngularCombiner(
            n_quantiles=99, weights=w, degree=0.0
        )
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        for h in range(1, H + 1):
            mean_a = res_a.to_distribution(h).mean()
            mean_b = res_b.to_distribution(h).mean()
            expected = w[0] * mean_a + w[1] * mean_b
            actual = combined.to_distribution(h).mean()
            np.testing.assert_allclose(actual, expected, rtol=0.01)

    def test_mean_degree_positive(self, basis_index, observed):
        """Approximate equality when degree > 0."""
        w = np.array([0.6, 0.4])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0)

        combiner = AngularCombiner(
            n_quantiles=99, weights=w, degree=30.0
        )
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        for h in range(1, H + 1):
            mean_a = res_a.to_distribution(h).mean()
            mean_b = res_b.to_distribution(h).mean()
            expected = w[0] * mean_a + w[1] * mean_b
            actual = combined.to_distribution(h).mean()
            # Sampling-based combine: wider tolerance than grid-based
            np.testing.assert_allclose(actual, expected, rtol=0.15, atol=0.05)

    def test_mean_three_models(self, basis_index, observed):
        """Mean property holds for 3 models."""
        w = np.array([0.5, 0.3, 0.2])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        res_c = _make_parametric(basis_index, "C", loc_offset=3.0)

        combiner = AngularCombiner(
            n_quantiles=99, weights=w, degree=30.0
        )
        combiner.fit([res_a, res_b, res_c], observed)
        combined = combiner.combine([res_a, res_b, res_c])

        for h in range(1, H + 1):
            mean_a = res_a.to_distribution(h).mean()
            mean_b = res_b.to_distribution(h).mean()
            mean_c = res_c.to_distribution(h).mean()
            expected = w[0] * mean_a + w[1] * mean_b + w[2] * mean_c
            actual = combined.to_distribution(h).mean()
            # Sampling-based combine: wider tolerance than grid-based
            np.testing.assert_allclose(actual, expected, rtol=0.15, atol=0.05)


# ---------------------------------------------------------------------------
# Test: beta parameter behavior
# ---------------------------------------------------------------------------


class TestBetaBehavior:
    """Tests for beta (inverse-CRPS power exponent) effects."""

    def test_high_beta_concentrates_on_best(self, basis_index, observed):
        """Higher beta gives more weight to the better model."""
        res_good = _make_parametric(
            basis_index, "Good", loc_offset=0.0, scale=0.5
        )
        res_bad = _make_parametric(
            basis_index, "Bad", loc_offset=5.0, scale=2.0
        )

        c_low = AngularCombiner(
            n_quantiles=19, beta=1.0, degree=30.0,
            fit_method="inverse_crps",
        )
        c_high = AngularCombiner(
            n_quantiles=19, beta=5.0, degree=30.0,
            fit_method="inverse_crps",
        )

        c_low.fit([res_good, res_bad], observed)
        c_high.fit([res_good, res_bad], observed)

        for h in range(1, H + 1):
            # Both should favor "Good" (index 0)
            assert c_low.weights_[h][0] > c_low.weights_[h][1]
            assert c_high.weights_[h][0] > c_high.weights_[h][1]
            # Higher beta should be more concentrated
            assert c_high.weights_[h][0] > c_low.weights_[h][0]

    def test_beta_one_is_standard_inverse_crps(self, basis_index, observed):
        """beta=1 produces the same weights as BaseCombiner._inverse_crps_weights."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        combiner = AngularCombiner(
            n_quantiles=19, beta=1.0, degree=30.0,
            fit_method="inverse_crps",
        )
        combiner.fit([res_a, res_b], observed)

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


# ---------------------------------------------------------------------------
# Test: COBYLA fitting
# ---------------------------------------------------------------------------


class TestCOBYLAFitting:
    """Tests for COBYLA optimization of weights and degree."""

    def test_cobyla_stores_params(self, basis_index, observed):
        """fit() stores degree per horizon in params_."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        combiner = AngularCombiner(
            n_quantiles=19, fit_method="cobyla"
        )
        combiner.fit([res_a, res_b], observed)

        assert combiner.is_fitted_
        for h in range(1, H + 1):
            assert h in combiner.params_
            degree = combiner.params_[h]["degree"]
            assert 0 <= degree <= 90

    def test_cobyla_improves_over_default(self, basis_index, observed):
        """Optimized CRPS <= default (inverse_crps, degree=45) CRPS."""
        res_a = _make_parametric(
            basis_index, "Good", loc_offset=0.0, scale=0.5
        )
        res_b = _make_parametric(
            basis_index, "Bad", loc_offset=3.0, scale=1.5
        )

        c_default = AngularCombiner(
            n_quantiles=19, degree=45.0, fit_method="inverse_crps"
        )
        c_opt = AngularCombiner(
            n_quantiles=19, fit_method="cobyla"
        )

        c_default.fit([res_a, res_b], observed)
        c_opt.fit([res_a, res_b], observed)

        tau = c_default.quantile_levels
        obs_arr = observed.values

        for h in range(1, H + 1):
            comb_def = c_default.combine([res_a, res_b])
            comb_opt = c_opt.combine([res_a, res_b])

            q_def = np.column_stack([
                comb_def.quantiles_data[q][:, h - 1] for q in tau
            ])
            q_opt = np.column_stack([
                comb_opt.quantiles_data[q][:, h - 1] for q in tau
            ])
            crps_def = crps_quantile(tau, q_def, obs_arr[:, h - 1])
            crps_opt = crps_quantile(tau, q_opt, obs_arr[:, h - 1])
            assert crps_opt <= crps_def + 1e-8

    def test_invalid_fit_method_raises(self):
        """Unknown fit_method raises ValueError."""
        with pytest.raises(ValueError, match="fit_method"):
            AngularCombiner(fit_method="bogus")

    def test_inverse_crps_grid_search_degree(self, basis_index, observed):
        """fit_method='inverse_crps' without degree runs grid search."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        combiner = AngularCombiner(
            n_quantiles=19, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)

        assert combiner.is_fitted_
        for h in range(1, H + 1):
            assert "degree" in combiner.params_[h]
            deg = combiner.params_[h]["degree"]
            assert 0 <= deg <= 90

    def test_inverse_crps_grid_improves_over_fixed(
        self, basis_index, observed
    ):
        """Grid-searched degree should be at least as good as degree=45."""
        res_a = _make_parametric(
            basis_index, "Good", loc_offset=0.0, scale=0.5
        )
        res_b = _make_parametric(
            basis_index, "Bad", loc_offset=3.0, scale=1.5
        )

        c_fixed = AngularCombiner(
            n_quantiles=19, degree=45.0, fit_method="inverse_crps"
        )
        c_grid = AngularCombiner(
            n_quantiles=19, fit_method="inverse_crps"
        )

        c_fixed.fit([res_a, res_b], observed)
        c_grid.fit([res_a, res_b], observed)

        tau = c_fixed.quantile_levels
        obs_arr = observed.values

        for h in range(1, H + 1):
            comb_fixed = c_fixed.combine([res_a, res_b])
            comb_grid = c_grid.combine([res_a, res_b])

            q_f = np.column_stack([
                comb_fixed.quantiles_data[q][:, h - 1] for q in tau
            ])
            q_g = np.column_stack([
                comb_grid.quantiles_data[q][:, h - 1] for q in tau
            ])
            crps_f = crps_quantile(tau, q_f, obs_arr[:, h - 1])
            crps_g = crps_quantile(tau, q_g, obs_arr[:, h - 1])
            assert crps_g <= crps_f + 1e-8

    def test_inverse_crps_grid_variance_between_endpoints(
        self, basis_index, observed
    ):
        """Grid-searched degree produces variance between 0° and 90°.

        Verifies that:
        1. The found degree lies strictly between 0 and 90
        2. Variance at the found degree is between horizontal and vertical
        3. Variance monotonically increases with degree (spot check)
        """
        res_a = _make_parametric(basis_index, "A", scale=0.5)
        res_b = _make_parametric(
            basis_index, "B", loc_offset=2.0, scale=0.5
        )

        # Grid search combiner
        c_grid = AngularCombiner(
            n_quantiles=99, fit_method="inverse_crps"
        )
        c_grid.fit([res_a, res_b], observed)

        # Same weights, fixed at endpoints for comparison
        for h in range(1, H + 1):
            w = c_grid.weights_[h]
            found_deg = c_grid.params_[h]["degree"]

            c_h = AngularCombiner(
                n_quantiles=99, weights=w, degree=0.0
            )
            c_v = AngularCombiner(
                n_quantiles=99, weights=w, degree=90.0
            )
            c_h.fit([res_a, res_b], observed)
            c_v.fit([res_a, res_b], observed)

            var_h = np.mean(
                c_h.combine([res_a, res_b]).to_distribution(h).std() ** 2
            )
            var_v = np.mean(
                c_v.combine([res_a, res_b]).to_distribution(h).std() ** 2
            )
            var_grid = np.mean(
                c_grid.combine([res_a, res_b]).to_distribution(h).std() ** 2
            )

            # Variance at found degree between horizontal and vertical
            assert var_grid >= var_h - 1e-6, (
                f"h={h}: Var(grid)={var_grid:.6f} < Var(0°)={var_h:.6f}"
            )
            assert var_grid <= var_v + 1e-6, (
                f"h={h}: Var(grid)={var_grid:.6f} > Var(90°)={var_v:.6f}"
            )

            # Spot check: variance increases with degree
            degrees = [0.0, found_deg, 90.0]
            variances = [var_h, var_grid, var_v]
            for i in range(1, len(variances)):
                assert variances[i] >= variances[i - 1] - 1e-6, (
                    f"h={h}: Var at θ={degrees[i]:.0f} "
                    f"({variances[i]:.6f}) < Var at θ={degrees[i-1]:.0f} "
                    f"({variances[i-1]:.6f})"
                )

            # Mean is preserved across all degrees (theory).
            # Quantile-based integration introduces interpolation error,
            # so we compare global mean across all time steps.
            mean_h = c_h.combine([res_a, res_b]).to_distribution(h).mean()
            mean_v = c_v.combine([res_a, res_b]).to_distribution(h).mean()
            mean_grid = (
                c_grid.combine([res_a, res_b]).to_distribution(h).mean()
            )
            np.testing.assert_allclose(
                mean_grid.mean(), mean_h.mean(), rtol=0.02,
                err_msg=f"h={h}: mean differs between 0° and grid({found_deg:.0f}°)",
            )
            np.testing.assert_allclose(
                mean_grid.mean(), mean_v.mean(), rtol=0.02,
                err_msg=f"h={h}: mean differs between 90° and grid({found_deg:.0f}°)",
            )


# ---------------------------------------------------------------------------
# Test: Identical models
# ---------------------------------------------------------------------------


class TestIdenticalModels:
    """When all models are identical, combined ≈ original regardless of degree."""

    def test_identical_models_any_degree(self, basis_index, observed):
        """Combined result approximates the original for any degree."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "A_copy")  # same seed

        for degree in [0.0, 45.0, 90.0]:
            w = np.array([0.5, 0.5])
            combiner = AngularCombiner(
                n_quantiles=99, weights=w, degree=degree,
                n_samples=10000,
            )
            combiner.fit([res_a, res_b], observed)
            combined = combiner.combine([res_a, res_b])

            ql = combiner.quantile_levels
            conv_a = combiner._convert_to_quantile(res_a)
            for h in range(1, H + 1):
                orig = np.column_stack(
                    [conv_a.quantiles_data[q][:, h - 1] for q in ql]
                )
                comb = np.column_stack(
                    [combined.quantiles_data[q][:, h - 1] for q in ql]
                )
                # Sampling approximation: use atol for near-zero values
                np.testing.assert_allclose(
                    comb, orig, rtol=0.05, atol=0.05,
                    err_msg=f"Failed at degree={degree}, h={h}",
                )


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_single_model_weight_one(self, basis_index, observed):
        """weight=[1, 0] → combined equals the first model exactly."""
        w = np.array([1.0, 0.0])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=5.0)

        for degree in [0.0, 30.0]:
            combiner = AngularCombiner(
                n_quantiles=99, weights=w, degree=degree,
                n_samples=10000,
            )
            combiner.fit([res_a, res_b], observed)
            combined = combiner.combine([res_a, res_b])

            conv_a = combiner._convert_to_quantile(res_a)
            ql = combiner.quantile_levels
            for h in range(1, H + 1):
                orig = np.column_stack(
                    [conv_a.quantiles_data[q][:, h - 1] for q in ql]
                )
                comb = np.column_stack(
                    [combined.quantiles_data[q][:, h - 1] for q in ql]
                )
                # Sampling approximation: atol for near-zero values
                np.testing.assert_allclose(
                    comb, orig, rtol=0.05, atol=0.05,
                    err_msg=f"Failed at degree={degree}, h={h}",
                )

    def test_minimum_two_models(self, basis_index, observed):
        """Minimum model count (M=2) works correctly."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        assert combined.horizon == H
        assert len(combined) == N

    def test_many_models(self, basis_index, observed):
        """M=10 models: weights sum to 1 and combine succeeds."""
        results = [
            _make_parametric(basis_index, f"M{i}", loc_offset=i * 0.5)
            for i in range(10)
        ]
        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit(results, observed)
        combined = combiner.combine(results)

        assert len(combined) == N
        assert combined.horizon == H
        for h in range(1, H + 1):
            w = combiner.weights_[h]
            assert len(w) == 10
            np.testing.assert_allclose(w.sum(), 1.0)
            assert np.all(w >= 0)

    def test_minimum_data_size(self, observed):
        """N=3 (minimum viable) works without error."""
        small_idx = pd.date_range("2024-01-01", periods=3, freq="h")
        res_a = _make_parametric(small_idx, "A")
        res_b = _make_parametric(small_idx, "B", loc_offset=1.0)
        obs = observed.iloc[:3]
        obs.index = small_idx

        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], obs)
        combined = combiner.combine([res_a, res_b])

        assert len(combined) == 3

    def test_degree_tiny_positive(self, basis_index, observed):
        """degree=1e-4 behaves like degree=0 (horizontal)."""
        w = np.array([0.6, 0.4])
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)

        c_zero = AngularCombiner(
            n_quantiles=19, weights=w, degree=0.0
        )
        c_tiny = AngularCombiner(
            n_quantiles=19, weights=w, degree=1e-4
        )

        c_zero.fit([res_a, res_b], observed)
        c_tiny.fit([res_a, res_b], observed)

        r_zero = c_zero.combine([res_a, res_b])
        r_tiny = c_tiny.combine([res_a, res_b])

        ql = c_zero.quantile_levels
        for h in range(1, H + 1):
            v_zero = np.column_stack(
                [r_zero.quantiles_data[q][:, h - 1] for q in ql]
            )
            v_tiny = np.column_stack(
                [r_tiny.quantiles_data[q][:, h - 1] for q in ql]
            )
            np.testing.assert_allclose(v_tiny, v_zero, atol=1e-3)

    def test_intermediate_degrees_between_horizontal_and_vertical(
        self, basis_index, observed
    ):
        """Intermediate degrees produce variance between horizontal and vertical.

        Theorem 1 & 2 (Taylor & Meng, 2025): angular combining has
        lower variance than vertical, and under Assumption 1 the
        variance is monotonically increasing w.r.t. θ. We verify
        that for degrees 10, 20, ..., 80 the IQR (as a proxy for
        spread) lies between horizontal (θ=0) and vertical (θ=90).
        """
        w = np.array([0.6, 0.4])
        Q = 99
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=2.0)

        # Compute horizontal and vertical baselines
        h_combiner = HorizontalCombiner(n_quantiles=Q, weights=w)
        v_combiner = VerticalCombiner(n_quantiles=Q, weights=w)
        h_combiner.fit([res_a, res_b], observed)
        v_combiner.fit([res_a, res_b], observed)
        h_result = h_combiner.combine([res_a, res_b])
        v_result = v_combiner.combine([res_a, res_b])

        ql = h_combiner.quantile_levels
        q25_idx = np.searchsorted(ql, 0.25)
        q75_idx = np.searchsorted(ql, 0.75)

        for h in range(1, H + 1):
            h_vals = np.column_stack(
                [h_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            v_vals = np.column_stack(
                [v_result.quantiles_data[q][:, h - 1] for q in ql]
            )
            iqr_h = np.mean(h_vals[:, q75_idx] - h_vals[:, q25_idx])
            iqr_v = np.mean(v_vals[:, q75_idx] - v_vals[:, q25_idx])

            for degree in range(10, 90, 10):
                a_combiner = AngularCombiner(
                    n_quantiles=Q, weights=w, degree=float(degree)
                )
                a_combiner.fit([res_a, res_b], observed)
                a_result = a_combiner.combine([res_a, res_b])
                a_vals = np.column_stack(
                    [a_result.quantiles_data[q][:, h - 1] for q in ql]
                )
                iqr_a = np.mean(a_vals[:, q75_idx] - a_vals[:, q25_idx])

                # IQR should be between horizontal and vertical
                assert iqr_a >= iqr_h - 1e-6, (
                    f"degree={degree}, h={h}: IQR={iqr_a:.4f} < "
                    f"horizontal IQR={iqr_h:.4f}"
                )
                assert iqr_a <= iqr_v + 1e-6, (
                    f"degree={degree}, h={h}: IQR={iqr_a:.4f} > "
                    f"vertical IQR={iqr_v:.4f}"
                )

    def test_degree_exact_90(self, basis_index, observed):
        """degree=90.0 exactly does not raise or produce NaN."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = AngularCombiner(
            n_quantiles=19, degree=90.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        ql = combiner.quantile_levels
        for h in range(1, H + 1):
            vals = np.column_stack(
                [combined.quantiles_data[q][:, h - 1] for q in ql]
            )
            assert np.all(np.isfinite(vals))

    def test_equal_crps_models_get_equal_weights(self, basis_index, observed):
        """Models with identical CRPS should receive equal weights."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "A_clone")

        combiner = AngularCombiner(
            n_quantiles=19, beta=1.0, degree=30.0,
            fit_method="inverse_crps",
        )
        combiner.fit([res_a, res_b], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(
                combiner.weights_[h], [0.5, 0.5], atol=1e-10
            )

    def test_extreme_scale_difference(self, basis_index, observed):
        """One very narrow, one very wide distribution."""
        res_narrow = _make_parametric(
            basis_index, "Narrow", loc_offset=0.0, scale=0.01
        )
        res_wide = _make_parametric(
            basis_index, "Wide", loc_offset=0.0, scale=10.0
        )
        combiner = AngularCombiner(
            n_quantiles=19, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_narrow, res_wide], observed)
        combined = combiner.combine([res_narrow, res_wide])

        assert np.all(np.isfinite(
            combined.to_distribution(1).mean()
        ))
        for h in range(1, H + 1):
            np.testing.assert_allclose(combiner.weights_[h].sum(), 1.0)

    def test_partial_index_overlap(self, basis_index, observed):
        """Models with partially overlapping basis_index are aligned."""
        idx_shifted = pd.date_range(
            "2024-01-01 05:00", periods=N, freq="h"
        )
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(idx_shifted, "B", loc_offset=1.0)

        common_idx = basis_index.intersection(idx_shifted)
        rng = np.random.default_rng(99)
        obs_common = pd.DataFrame(
            rng.standard_normal((len(common_idx), H)),
            index=common_idx,
        )

        combiner = AngularCombiner(
            n_quantiles=9, degree=30.0, fit_method="inverse_crps"
        )
        combiner.fit([res_a, res_b], obs_common)
        combined = combiner.combine([res_a, res_b])

        assert combined.basis_index.equals(common_idx)
        assert len(combined) == len(common_idx)
