"""Tests for BaseCombiner and EqualWeightCombiner."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    ParametricForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)
from src.models.combining.base import BaseCombiner
from src.models.combining.equal_weight import EqualWeightCombiner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, H, N_SAMPLES = 20, 3, 50


@pytest.fixture
def basis_index():
    return pd.date_range("2024-01-01", periods=N, freq="h")


@pytest.fixture
def basis_index_shifted():
    """Overlaps with basis_index by 15 points (shifted by 5)."""
    return pd.date_range("2024-01-01 05:00", periods=N, freq="h")


def _make_parametric(basis_index, model_name, loc_offset=0.0):
    """Create a ParametricForecastResult with known values."""
    n = len(basis_index)
    rng = np.random.default_rng(42 + int(loc_offset))
    return ParametricForecastResult(
        dist_name="normal",
        params={
            "loc": rng.standard_normal((n, H)) + loc_offset,
            "scale": np.ones((n, H)) * 0.5,
        },
        basis_index=basis_index,
        model_name=model_name,
    )


def _make_sample(basis_index, model_name, seed=0):
    """Create a SampleForecastResult with known values."""
    n = len(basis_index)
    rng = np.random.default_rng(seed)
    return SampleForecastResult(
        samples=rng.standard_normal((n, N_SAMPLES, H)),
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
# Test: _validate_results
# ---------------------------------------------------------------------------


class TestValidateResults:
    """Tests for input validation."""

    def test_single_model_raises(self, basis_index):
        res = _make_parametric(basis_index, "A")
        combiner = EqualWeightCombiner()
        with pytest.raises(ValueError, match="At least 2"):
            combiner._validate_results([res])

    def test_empty_list_raises(self):
        combiner = EqualWeightCombiner()
        with pytest.raises(ValueError, match="At least 2"):
            combiner._validate_results([])

    def test_horizon_mismatch_raises(self, basis_index):
        res_a = _make_parametric(basis_index, "A")
        # Create result with different horizon
        res_b = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": np.zeros((N, 5)),
                "scale": np.ones((N, 5)),
            },
            basis_index=basis_index,
            model_name="B",
        )
        combiner = EqualWeightCombiner()
        with pytest.raises(ValueError, match="Horizon mismatch"):
            combiner._validate_results([res_a, res_b])

    def test_valid_results_pass(self, basis_index):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = EqualWeightCombiner()
        combiner._validate_results([res_a, res_b])  # no error

    def test_observed_with_nan_raises(self, basis_index):
        """observed with NaN values should raise at fit time."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        rng = np.random.default_rng(99)
        obs_nan = pd.DataFrame(
            rng.standard_normal((N, H)),
            index=basis_index,
        )
        obs_nan.iloc[5, 1] = np.nan
        combiner = EqualWeightCombiner(n_quantiles=9)
        with pytest.raises(ValueError, match="NaN"):
            combiner.fit([res_a, res_b], obs_nan)

    def test_observed_wrong_columns_raises(self, basis_index):
        """observed with wrong number of columns should raise."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        rng = np.random.default_rng(99)
        # H+1 columns instead of H
        obs_wrong = pd.DataFrame(
            rng.standard_normal((N, H + 1)),
            index=basis_index,
        )
        combiner = EqualWeightCombiner(n_quantiles=9)
        with pytest.raises(ValueError, match="columns"):
            combiner.fit([res_a, res_b], obs_wrong)


# ---------------------------------------------------------------------------
# Test: _align_results
# ---------------------------------------------------------------------------


class TestAlignResults:
    """Tests for basis_index alignment."""

    def test_same_index_no_copy(self, basis_index):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = EqualWeightCombiner()
        aligned = combiner._align_results([res_a, res_b])
        # Same objects returned when indices match
        assert aligned[0] is res_a
        assert aligned[1] is res_b

    def test_different_index_aligned(self, basis_index, basis_index_shifted):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index_shifted, "B", loc_offset=1.0)
        combiner = EqualWeightCombiner()
        aligned = combiner._align_results([res_a, res_b])
        assert aligned[0].basis_index.equals(aligned[1].basis_index)
        # Common overlap: 15 points
        assert len(aligned[0]) == 15

    def test_no_overlap_raises(self, basis_index):
        idx_far = pd.date_range("2025-01-01", periods=N, freq="h")
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(idx_far, "B", loc_offset=1.0)
        combiner = EqualWeightCombiner()
        with pytest.raises(ValueError, match="empty"):
            combiner._align_results([res_a, res_b])


# ---------------------------------------------------------------------------
# Test: EqualWeightCombiner end-to-end
# ---------------------------------------------------------------------------


class TestEqualWeightCombiner:
    """End-to-end tests for EqualWeightCombiner."""

    def test_fit_sets_equal_weights(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "ModelA")
        res_b = _make_parametric(basis_index, "ModelB", loc_offset=1.0)
        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_a, res_b], observed)

        assert combiner.is_fitted_
        for h in range(1, H + 1):
            w = combiner.weights_[h]
            np.testing.assert_allclose(w, [0.5, 0.5])

    def test_combine_output_shape(self, basis_index, observed):
        Q = 9
        res_a = _make_parametric(basis_index, "ModelA")
        res_b = _make_parametric(basis_index, "ModelB", loc_offset=1.0)
        combiner = EqualWeightCombiner(n_quantiles=Q)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert len(combined.quantile_levels) == Q
        assert combined.horizon == H
        assert combined.basis_index.equals(basis_index)

    def test_combine_before_fit_raises(self, basis_index):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = EqualWeightCombiner()
        with pytest.raises(RuntimeError, match="fit"):
            combiner.combine([res_a, res_b])

    def test_combine_reordered_models_raises(self, basis_index, observed):
        """Reordering models between fit and combine should raise."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_a, res_b], observed)

        with pytest.raises(ValueError, match="Model names/order mismatch"):
            combiner.combine([res_b, res_a])

    def test_combine_swapped_model_raises(self, basis_index, observed):
        """Swapping a model with different name should raise."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        res_c = _make_parametric(basis_index, "C", loc_offset=2.0)
        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_a, res_b], observed)

        with pytest.raises(ValueError, match="Model names/order mismatch"):
            combiner.combine([res_a, res_c])

    def test_combine_fewer_models_raises(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        res_c = _make_parametric(basis_index, "C", loc_offset=2.0)
        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_a, res_b, res_c], observed)

        with pytest.raises(ValueError, match="Expected 3 models"):
            combiner.combine([res_a, res_b])

    def test_combine_more_models_raises(self, basis_index, observed):
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        res_c = _make_parametric(basis_index, "C", loc_offset=2.0)
        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_a, res_b], observed)

        with pytest.raises(ValueError, match="Expected 2 models"):
            combiner.combine([res_a, res_b, res_c])

    def test_combine_is_mean_of_quantiles(self, basis_index, observed):
        """Verify combined quantiles equal the mean of individual quantiles."""
        Q = 9
        res_a = _make_parametric(basis_index, "ModelA")
        res_b = _make_parametric(basis_index, "ModelB", loc_offset=2.0)
        combiner = EqualWeightCombiner(n_quantiles=Q)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        # Compute expected from converted quantile arrays (same as combiner sees)
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
            expected = 0.5 * q_a + 0.5 * q_b
            actual = np.column_stack(
                [combined.quantiles_data[q][:, h - 1] for q in ql]
            )
            np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_combine_with_misaligned_index(
        self, basis_index, basis_index_shifted, observed
    ):
        """fit and combine work with different basis_index lengths."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index_shifted, "B", loc_offset=1.0)

        # observed must cover the common index
        common_idx = basis_index.intersection(basis_index_shifted)
        rng = np.random.default_rng(99)
        obs = pd.DataFrame(
            rng.standard_normal((len(common_idx), H)),
            index=common_idx,
        )

        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_a, res_b], obs)
        combined = combiner.combine([res_a, res_b])

        assert combined.basis_index.equals(common_idx)
        assert len(combined) == len(common_idx)

    def test_three_models(self, basis_index, observed):
        """Works with 3 models, weights are 1/3 each."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        res_c = _make_parametric(basis_index, "C", loc_offset=2.0)
        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_a, res_b, res_c], observed)

        for h in range(1, H + 1):
            np.testing.assert_allclose(
                combiner.weights_[h],
                [1 / 3, 1 / 3, 1 / 3],
            )

        combined = combiner.combine([res_a, res_b, res_c])
        assert len(combined) == N
        assert len(combined.quantile_levels) == 9
        assert combined.horizon == H

    def test_mixed_result_types(self, basis_index, observed):
        """Parametric and Sample results can be combined."""
        res_param = _make_parametric(basis_index, "Param")
        res_sample = _make_sample(basis_index, "Sample", seed=7)
        combiner = EqualWeightCombiner(n_quantiles=9)
        combiner.fit([res_param, res_sample], observed)
        combined = combiner.combine([res_param, res_sample])

        assert isinstance(combined, QuantileForecastResult)
        assert len(combined) == N
        assert len(combined.quantile_levels) == 9
        assert combined.horizon == H

    def test_to_distribution_on_combined(self, basis_index, observed):
        """Combined result supports to_distribution()."""
        res_a = _make_parametric(basis_index, "A")
        res_b = _make_parametric(basis_index, "B", loc_offset=1.0)
        combiner = EqualWeightCombiner(n_quantiles=19)
        combiner.fit([res_a, res_b], observed)
        combined = combiner.combine([res_a, res_b])

        dist = combined.to_distribution(1)
        assert dist.mean().shape == (N,)
        assert dist.ppf([0.1, 0.5, 0.9]).shape == (N, 3)

    def test_duplicate_model_names_raises(self, basis_index, observed):
        """Duplicate model names should be rejected at fit time."""
        res_a = _make_parametric(basis_index, "Same")
        res_b = _make_parametric(basis_index, "Same", loc_offset=1.0)
        combiner = EqualWeightCombiner(n_quantiles=9)
        with pytest.raises(ValueError, match="Duplicate model_name"):
            combiner.fit([res_a, res_b], observed)

    def test_empty_model_name_raises(self, basis_index, observed):
        """Empty model_name should be rejected at fit time."""
        res_a = _make_parametric(basis_index, "A")
        res_b = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": np.zeros((N, H)),
                "scale": np.ones((N, H)),
            },
            basis_index=basis_index,
            model_name="",
        )
        combiner = EqualWeightCombiner(n_quantiles=9)
        with pytest.raises(ValueError, match="empty model_name"):
            combiner.fit([res_a, res_b], observed)
