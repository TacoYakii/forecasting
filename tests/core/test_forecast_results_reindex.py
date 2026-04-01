"""Tests for BaseForecastResult.reindex() and model_name propagation."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    ParametricForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basis_index():
    return pd.date_range("2024-01-01", periods=10, freq="h")


@pytest.fixture
def parametric_result(basis_index):
    N, H = len(basis_index), 3
    return ParametricForecastResult(
        dist_name="normal",
        params={
            "loc": np.arange(N * H, dtype=float).reshape(N, H),
            "scale": np.ones((N, H)),
        },
        basis_index=basis_index,
        model_name="TestModel",
    )


@pytest.fixture
def quantile_result(basis_index):
    N, H = len(basis_index), 3
    return QuantileForecastResult(
        quantiles_data={
            0.1: np.random.default_rng(0).standard_normal((N, H)),
            0.5: np.random.default_rng(1).standard_normal((N, H)),
            0.9: np.random.default_rng(2).standard_normal((N, H)),
        },
        basis_index=basis_index,
        model_name="QuantileModel",
    )


@pytest.fixture
def sample_result(basis_index):
    N, S, H = len(basis_index), 50, 3
    return SampleForecastResult(
        samples=np.random.default_rng(0).standard_normal((N, S, H)),
        basis_index=basis_index,
        model_name="SampleModel",
    )


# ---------------------------------------------------------------------------
# model_name
# ---------------------------------------------------------------------------

class TestModelName:
    """model_name attribute on ForecastResult."""

    def test_default_model_name(self):
        idx = pd.RangeIndex(2)
        result = ParametricForecastResult(
            dist_name="normal",
            params={"loc": np.zeros((2, 1)), "scale": np.ones((2, 1))},
            basis_index=idx,
        )
        assert result.model_name == ""

    def test_explicit_model_name(self, parametric_result):
        assert parametric_result.model_name == "TestModel"

    def test_model_name_on_quantile(self, quantile_result):
        assert quantile_result.model_name == "QuantileModel"

    def test_model_name_on_sample(self, sample_result):
        assert sample_result.model_name == "SampleModel"


# ---------------------------------------------------------------------------
# reindex — ParametricForecastResult
# ---------------------------------------------------------------------------

class TestParametricReindex:
    """ParametricForecastResult.reindex()."""

    def test_subset(self, parametric_result, basis_index):
        sub_idx = basis_index[2:5]
        reindexed = parametric_result.reindex(sub_idx)

        assert isinstance(reindexed, ParametricForecastResult)
        assert len(reindexed) == 3
        assert reindexed.basis_index.equals(sub_idx)
        assert reindexed.dist_name == parametric_result.dist_name
        assert reindexed.model_name == "TestModel"

        for k in parametric_result.params:
            np.testing.assert_array_equal(
                reindexed.params[k],
                parametric_result.params[k][2:5, :],
            )

    def test_non_contiguous(self, parametric_result, basis_index):
        sub_idx = basis_index[[0, 3, 7]]
        reindexed = parametric_result.reindex(sub_idx)

        assert len(reindexed) == 3
        for k in parametric_result.params:
            np.testing.assert_array_equal(
                reindexed.params[k],
                parametric_result.params[k][[0, 3, 7], :],
            )

    def test_full_index(self, parametric_result, basis_index):
        reindexed = parametric_result.reindex(basis_index)
        assert len(reindexed) == len(basis_index)
        for k in parametric_result.params:
            np.testing.assert_array_equal(
                reindexed.params[k],
                parametric_result.params[k],
            )

    def test_missing_index_raises(self, parametric_result):
        bad_idx = pd.DatetimeIndex(["2099-01-01"])
        with pytest.raises(KeyError):
            parametric_result.reindex(bad_idx)


# ---------------------------------------------------------------------------
# reindex — QuantileForecastResult
# ---------------------------------------------------------------------------

class TestQuantileReindex:
    """QuantileForecastResult.reindex()."""

    def test_subset(self, quantile_result, basis_index):
        sub_idx = basis_index[1:4]
        reindexed = quantile_result.reindex(sub_idx)

        assert isinstance(reindexed, QuantileForecastResult)
        assert len(reindexed) == 3
        assert reindexed.model_name == "QuantileModel"
        assert reindexed.quantile_levels == quantile_result.quantile_levels

        for q in quantile_result.quantile_levels:
            np.testing.assert_array_equal(
                reindexed.quantiles_data[q],
                quantile_result.quantiles_data[q][1:4, :],
            )

    def test_missing_index_raises(self, quantile_result):
        bad_idx = pd.DatetimeIndex(["2099-01-01"])
        with pytest.raises(KeyError):
            quantile_result.reindex(bad_idx)


# ---------------------------------------------------------------------------
# reindex — SampleForecastResult
# ---------------------------------------------------------------------------

class TestSampleReindex:
    """SampleForecastResult.reindex()."""

    def test_subset(self, sample_result, basis_index):
        sub_idx = basis_index[5:8]
        reindexed = sample_result.reindex(sub_idx)

        assert isinstance(reindexed, SampleForecastResult)
        assert len(reindexed) == 3
        assert reindexed.model_name == "SampleModel"
        assert reindexed.n_samples == sample_result.n_samples

        np.testing.assert_array_equal(
            reindexed.samples,
            sample_result.samples[5:8, :, :],
        )

    def test_missing_index_raises(self, sample_result):
        bad_idx = pd.DatetimeIndex(["2099-01-01"])
        with pytest.raises(KeyError):
            sample_result.reindex(bad_idx)
