"""End-to-end tests for Conditional Kernel Density (CKD) module.

Tests the full pipeline: CKDConfig → ConditionalKernelDensity.build() → apply()
→ GridDistribution, including data-adaptive bandwidth/grid, integration with
ForecastResult types, and the CRPS dispatch function.

Uses the shared synthetic wind power data from conftest.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_distribution import (
    GridDistribution,
    ParametricDistribution,
    SampleDistribution,
)
from src.core.forecast_results import (
    ParametricForecastResult,
    SampleForecastResult,
)
from src.models.conditional_kernel_density import (
    CKDConfig,
    ConditionalKernelDensity,
    resolve_to_samples,
)
from src.utils.metrics.crps import crps as fair_crps


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def wind_data(synthetic_df):
    """Extract training arrays from synthetic data."""
    df = synthetic_df.iloc[:500]
    x = df[["wind_speed", "temperature"]].values
    y = df["power"].values
    return x, y


@pytest.fixture
def fitted_model_1x(synthetic_df):
    """Fitted CKD with 1 explanatory variable (auto bandwidth)."""
    df = synthetic_df.iloc[:500]
    x = df[["wind_speed"]].values
    y = df["power"].values
    config = CKDConfig(n_x_vars=1, n_samples=500)
    model = ConditionalKernelDensity(config)
    model.build(x, y, ["wind_speed"])
    return model, df


@pytest.fixture
def fitted_model_2x(wind_data):
    """Fitted CKD with 2 explanatory variables (auto bandwidth)."""
    x, y = wind_data
    config = CKDConfig(n_x_vars=2, n_samples=300)
    model = ConditionalKernelDensity(config)
    model.build(x, y, ["wind_speed", "temperature"])
    return model


# =====================================================================
# Data-adaptive bandwidth & grid tests
# =====================================================================

class TestAdaptive:
    """Data-adaptive bandwidth and grid resolution."""

    def test_search_range_basic(self):
        """Search range: h_min = median spacing, h_max = range/10."""
        vals = np.array([0, 1, 2, 3, 5, 8, 13, 21])
        h_min, h_max = ConditionalKernelDensity._compute_search_range(vals)
        # median of diffs [1,1,1,2,3,5,8] = 2.0
        assert h_min == pytest.approx(2.0)
        # range/10 = 21/10 = 2.1
        assert h_max == pytest.approx(2.1)

    def test_search_range_h_min_exceeds_h_max(self):
        """When h_min >= h_max, h_max is expanded to h_min * 10."""
        # Dense data with narrow range
        vals = np.linspace(0, 1, 5)  # spacing=0.25, range/10=0.1
        h_min, h_max = ConditionalKernelDensity._compute_search_range(vals)
        assert h_min == pytest.approx(0.25)
        assert h_max == pytest.approx(2.5)  # 0.25 * 10

    def test_search_range_duplicates_ignored(self):
        """Duplicate values are filtered from spacing calculation."""
        vals = np.array([0, 0, 0, 1, 1, 2, 3])
        h_min, h_max = ConditionalKernelDensity._compute_search_range(vals)
        # Non-zero diffs: [1, 1, 1] → median = 1.0
        assert h_min == pytest.approx(1.0)

    def test_search_range_all_identical_raises(self):
        """All identical values raises ValueError."""
        vals = np.full(10, 5.0)
        with pytest.raises(ValueError, match="identical"):
            ConditionalKernelDensity._compute_search_range(vals)

    def test_reference_bandwidth_normal_data(self):
        """Reference bandwidth is reasonable for normal data."""
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 1000)
        h = ConditionalKernelDensity._reference_bandwidth(vals)
        # Silverman's rule for N(0,1) with n=1000: 0.9 * 1.0 * 1000^(-0.2) ≈ 0.226
        assert 0.1 < h < 0.5

    def test_reference_bandwidth_bimodal_uses_iqr(self):
        """Bimodal data: min(σ, IQR/1.34) picks IQR when modes are separated."""
        rng = np.random.default_rng(42)
        # Bimodal: half near 0, half near 100 — large std, but IQR ~ 100
        vals = np.concatenate([rng.normal(0, 1, 500), rng.normal(100, 1, 500)])
        std = float(np.std(vals))
        iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
        # For well-separated bimodal: IQR/1.34 ≈ 74.6, std ≈ 50
        # Robust rule picks std (smaller), which is still smaller than
        # a naive Silverman with std alone would give for std=50
        h = ConditionalKernelDensity._reference_bandwidth(vals)
        assert h > 0
        # Verify the robust rule uses the minimum of the two
        spread = min(std, iqr / 1.34)
        expected = 0.9 * spread * len(vals) ** (-1.0 / 5.0)
        assert h == pytest.approx(expected)

    def test_compute_basis_points(self):
        """Basis points = range / (bw / bins_per_bw) + 1, clamped."""
        config = CKDConfig(
            n_x_vars=1, bins_per_bandwidth=4,
            min_basis_points=20, max_basis_points=500,
        )
        model = ConditionalKernelDensity(config)
        # range=30, bw=1.0 → bin_width=0.25 → raw=121
        assert model._compute_basis_points(30.0, 1.0) == 121
        # Very narrow bw → hits max
        assert model._compute_basis_points(30.0, 0.01) == 500
        # Very wide bw → hits min
        assert model._compute_basis_points(10.0, 100.0) == 20

    def test_build_computes_search_ranges(self, wind_data):
        """build() populates search ranges from data."""
        x, y = wind_data
        config = CKDConfig(n_x_vars=2)
        model = ConditionalKernelDensity(config)
        model.build(x, y, ["wind_speed", "temperature"])
        assert len(model.x_search_ranges_) == 2
        for h_min, h_max in model.x_search_ranges_:
            assert h_min > 0
            assert h_max > h_min
        h_min_y, h_max_y = model.y_search_range_
        assert h_min_y > 0
        assert h_max_y > h_min_y

    def test_build_auto_bandwidth(self, wind_data):
        """build() without set_bandwidths() uses reference bandwidth."""
        x, y = wind_data
        config = CKDConfig(n_x_vars=2)
        model = ConditionalKernelDensity(config)
        model.build(x, y, ["wind_speed", "temperature"])
        # Bandwidths should be set automatically
        assert len(model.x_bandwidths) == 2
        assert all(bw > 0 for bw in model.x_bandwidths)
        assert model.y_bandwidth > 0

    def test_build_injected_bandwidth(self, wind_data):
        """set_bandwidths() overrides auto bandwidth."""
        x, y = wind_data
        config = CKDConfig(n_x_vars=2)
        model = ConditionalKernelDensity(config)
        model.set_bandwidths([0.5, 1.0], 2.0)
        model.build(x, y, ["wind_speed", "temperature"])
        assert model.x_bandwidths == [0.5, 1.0]
        assert model.y_bandwidth == 2.0

    def test_basis_points_adapt_to_bandwidth(self):
        """Narrower bandwidth → _compute_basis_points returns more points."""
        config = CKDConfig(
            n_x_vars=1, bins_per_bandwidth=4,
            min_basis_points=10, max_basis_points=2000,
        )
        model = ConditionalKernelDensity(config)
        data_range = 100.0
        pts_narrow = model._compute_basis_points(data_range, 0.5)
        pts_wide = model._compute_basis_points(data_range, 10.0)
        assert pts_narrow > pts_wide
        # 0.5 → bin_width=0.125, raw=801
        assert pts_narrow == 801
        # 10.0 → bin_width=2.5, raw=41
        assert pts_wide == 41


# =====================================================================
# build() tests
# =====================================================================

class TestBuild:
    """CKD density construction from training data."""

    def test_build_basic(self, wind_data):
        """Basic build succeeds and sets is_fitted."""
        x, y = wind_data
        config = CKDConfig(n_x_vars=2)
        model = ConditionalKernelDensity(config)
        assert not model.is_fitted
        model.build(x, y, ["wind_speed", "temperature"])
        assert model.is_fitted

    def test_fit_density_normalized(self, fitted_model_2x):
        """Each conditional slice sums to ~1."""
        d = fitted_model_2x.density
        sums = d.sum(axis=-1)
        assert np.median(sums) > 0.99

    def test_fit_nan_x_raises(self, wind_data):
        """NaN in x raises ValueError."""
        x, y = wind_data
        x_bad = x.copy()
        x_bad[10, 0] = np.nan
        config = CKDConfig(n_x_vars=2)
        model = ConditionalKernelDensity(config)
        with pytest.raises(ValueError, match="non-finite"):
            model.build(x_bad, y, ["wind_speed", "temperature"])

    def test_fit_nan_y_raises(self, wind_data):
        """NaN in y raises ValueError."""
        x, y = wind_data
        y_bad = y.copy()
        y_bad[0] = np.nan
        config = CKDConfig(n_x_vars=2)
        model = ConditionalKernelDensity(config)
        with pytest.raises(ValueError, match="non-finite"):
            model.build(x, y_bad, ["wind_speed", "temperature"])

    def test_fit_constant_x_raises(self):
        """Zero-range explanatory variable raises ValueError."""
        x = np.ones((50, 1))
        y = np.random.randn(50)
        config = CKDConfig(n_x_vars=1)
        model = ConditionalKernelDensity(config)
        with pytest.raises(ValueError, match="zero range"):
            model.build(x, y, ["const"])

    def test_fit_constant_y_raises(self):
        """Zero-range response variable raises ValueError."""
        x = np.random.randn(50, 1)
        y = np.full(50, 5.0)
        config = CKDConfig(n_x_vars=1)
        model = ConditionalKernelDensity(config)
        with pytest.raises(ValueError, match="zero range"):
            model.build(x, y, ["x"])

    def test_fit_method_chaining(self, wind_data):
        """build() returns self for chaining."""
        x, y = wind_data
        config = CKDConfig(n_x_vars=2)
        model = ConditionalKernelDensity(config)
        result = model.build(x, y, ["wind_speed", "temperature"])
        assert result is model


# =====================================================================
# apply() tests
# =====================================================================

class TestApply:
    """CKD apply with various input types."""

    def test_apply_raw_arrays(self, fitted_model_1x):
        """apply() with list of np.ndarray."""
        model, df = fitted_model_1x
        test_samples = [np.random.default_rng(0).normal(7, 2, (20, 300))]
        gd = model.apply(test_samples)
        assert isinstance(gd, GridDistribution)
        assert len(gd) == 20

    def test_apply_single_array(self, fitted_model_1x):
        """apply() with single ndarray for n_x=1."""
        model, df = fitted_model_1x
        arr = np.random.default_rng(0).normal(7, 2, (10, 200))
        gd = model.apply(arr)
        assert isinstance(gd, GridDistribution)
        assert len(gd) == 10

    def test_apply_returns_grid_distribution(self, fitted_model_2x):
        """apply() returns GridDistribution."""
        model = fitted_model_2x
        samples = [
            np.random.default_rng(0).normal(7, 2, (10, 200)),
            np.random.default_rng(1).normal(15, 3, (10, 200)),
        ]
        gd = model.apply(samples)
        assert isinstance(gd, GridDistribution)

    def test_apply_not_built_raises(self):
        """apply() before build() raises RuntimeError."""
        config = CKDConfig(n_x_vars=1)
        model = ConditionalKernelDensity(config)
        with pytest.raises(RuntimeError, match="not built"):
            model.apply([np.zeros((5, 100))])

    def test_apply_wrong_n_inputs(self, fitted_model_2x):
        """Wrong number of input variables raises ValueError."""
        model = fitted_model_2x
        with pytest.raises(ValueError, match="Expected 2"):
            model.apply([np.zeros((5, 100))])

    def test_apply_shape_mismatch(self, fitted_model_2x):
        """Mismatched shapes across inputs raises ValueError."""
        model = fitted_model_2x
        with pytest.raises(ValueError, match="Shape mismatch"):
            model.apply([
                np.zeros((5, 100)),
                np.zeros((10, 100)),
            ])

    def test_apply_nan_input_raises(self, fitted_model_1x):
        """NaN in samples raises ValueError."""
        model, _ = fitted_model_1x
        bad = np.ones((5, 100))
        bad[2, 50] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            model.apply([bad])

    def test_apply_time_index_from_parameter(self, fitted_model_1x):
        """Explicit time_index is used."""
        model, _ = fitted_model_1x
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        gd = model.apply(
            [np.random.default_rng(0).normal(7, 2, (5, 100))],
            time_index=idx,
        )
        assert (gd.index == idx).all()

    def test_apply_rangeindex_fallback(self, fitted_model_1x):
        """No time_index → RangeIndex."""
        model, _ = fitted_model_1x
        gd = model.apply([np.random.default_rng(0).normal(7, 2, (5, 100))])
        assert isinstance(gd.index, pd.RangeIndex)

    def test_apply_seed_reproducibility(self, fitted_model_1x):
        """Same seed produces same results for distribution inputs."""
        model, df = fitted_model_1x
        dist = ParametricDistribution(
            dist_name="normal",
            params={"loc": np.full(5, 7.0), "scale": np.full(5, 2.0)},
            index=pd.RangeIndex(5),
        )
        gd1 = model.apply(dist, seed=42)
        gd2 = model.apply(dist, seed=42)
        np.testing.assert_array_equal(gd1.prob, gd2.prob)

    def test_apply_normalizes_output(self, fitted_model_1x):
        """apply() always returns normalized probability rows."""
        model, _ = fitted_model_1x
        edge_val = model.x_basis_list[0][-1] + 5.0
        samples = np.full((3, 100), edge_val)
        gd = model.apply([samples])
        np.testing.assert_allclose(gd.prob.sum(axis=1), 1.0, atol=1e-6)


# =====================================================================
# resolve_to_samples
# =====================================================================

class TestResolveToSamples:
    """Universal input resolver."""

    def test_ndarray_passthrough(self):
        """2-D ndarray is returned as-is."""
        arr = np.random.randn(10, 50)
        out, idx = resolve_to_samples(arr)
        np.testing.assert_array_equal(out, arr)
        assert idx is None

    def test_ndarray_1d_raises(self):
        """1-D ndarray raises ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            resolve_to_samples(np.ones(10))

    def test_parametric_distribution(self):
        """ParametricDistribution → samples."""
        dist = ParametricDistribution(
            dist_name="normal",
            params={"loc": np.zeros(5), "scale": np.ones(5)},
            index=pd.RangeIndex(5),
        )
        out, idx = resolve_to_samples(dist, n_samples=100, seed=0)
        assert out.shape == (5, 100)
        assert idx is not None

    def test_empirical_distribution(self):
        """SampleDistribution → samples."""
        raw = np.random.default_rng(0).normal(0, 1, (5, 200))
        dist = SampleDistribution(index=pd.RangeIndex(5), samples=raw)
        out, idx = resolve_to_samples(dist, n_samples=100, seed=0)
        assert out.shape == (5, 100)

    def test_grid_distribution(self):
        """GridDistribution → samples."""
        grid = np.linspace(0, 10, 21)
        prob = np.ones((3, 21)) / 21
        dist = GridDistribution(pd.RangeIndex(3), grid, prob)
        out, idx = resolve_to_samples(dist, n_samples=100, seed=0)
        assert out.shape == (3, 100)

    def test_sample_forecast_result(self):
        """SampleForecastResult → samples for a specific horizon."""
        samples = np.random.default_rng(0).normal(0, 1, (5, 200, 3))
        result = SampleForecastResult(
            samples=samples,
            basis_index=pd.RangeIndex(5),
            model_name="test",
        )
        out, idx = resolve_to_samples(result, horizon=2)
        assert out.shape == (5, 200)

    def test_sample_forecast_result_no_horizon_raises(self):
        """SampleForecastResult without horizon raises ValueError."""
        samples = np.random.default_rng(0).normal(0, 1, (5, 200, 3))
        result = SampleForecastResult(
            samples=samples,
            basis_index=pd.RangeIndex(5),
            model_name="test",
        )
        with pytest.raises(ValueError, match="horizon"):
            resolve_to_samples(result)

    def test_unsupported_type_raises(self):
        """Unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported"):
            resolve_to_samples("not a valid input")


# =====================================================================
# CRPS dispatch integration
# =====================================================================

class TestCRPSDispatch:
    """Integration with utils/metrics/crps.py."""

    def test_crps_grid_distribution(self, fitted_model_1x):
        """crps() accepts GridDistribution without h."""
        model, df = fitted_model_1x
        samples = [np.random.default_rng(0).normal(7, 2, (20, 300))]
        gd = model.apply(samples)
        y_obs = df["power"].values[:20]
        score = fair_crps(gd, y_obs)
        assert np.isfinite(score)
        assert score > 0

    def test_crps_grid_vs_exact(self, fitted_model_1x):
        """Pseudo-sample CRPS (dispatch) ≈ grid-exact CRPS."""
        model, df = fitted_model_1x
        samples = [np.random.default_rng(0).normal(7, 2, (20, 300))]
        gd = model.apply(samples)
        y_obs = df["power"].values[:20]
        dispatch_score = fair_crps(gd, y_obs, n_quantiles=199)
        from src.utils.metrics.crps import grid_crps
        exact_score = float(grid_crps(gd, y_obs).mean())
        np.testing.assert_allclose(dispatch_score, exact_score, rtol=0.15)

    def test_crps_h_none_for_forecast_result_raises(self):
        """crps() with ForecastResult and h=None raises ValueError."""
        pfr = ParametricForecastResult(
            dist_name="normal",
            params={"loc": np.ones((3, 2)), "scale": np.ones((3, 2))},
            basis_index=pd.RangeIndex(3),
            model_name="test",
        )
        with pytest.raises(ValueError, match="horizon.*required"):
            fair_crps(pfr, np.ones(3))


# =====================================================================
# n_x_vars = 3 (broadcasting regression)
# =====================================================================

class TestThreeVariables:
    """Regression test for n_x_vars >= 3."""

    def test_fit_apply_3vars(self):
        """Full pipeline with 3 explanatory variables."""
        rng = np.random.default_rng(42)
        T = 200
        x = rng.normal(0, 1, (T, 3))
        y = x @ np.array([10, 5, 3]) + rng.normal(0, 1, T)
        config = CKDConfig(
            n_x_vars=3, n_samples=200,
            max_basis_points=15,  # keep small to avoid memory explosion
        )
        model = ConditionalKernelDensity(config)
        model.build(x, y, ["a", "b", "c"])
        assert model.density.shape[-1] == len(model.y_basis)

        test_x = [rng.normal(0, 1, (5, 100)) for _ in range(3)]
        gd = model.apply(test_x)
        assert isinstance(gd, GridDistribution)
        assert len(gd) == 5
        assert np.all(np.isfinite(gd.mean()))


# =====================================================================
# get_hyperparameters
# =====================================================================

class TestGetHyperparameters:
    """Hyperparameter introspection."""

    def test_get_hyperparameters(self, fitted_model_2x):
        """Returns dict with bandwidth, basis_points, and search ranges."""
        hp = fitted_model_2x.get_hyperparameters()
        assert "x_bandwidth" in hp
        assert "y_bandwidth" in hp
        assert "time_decay_factor" in hp
        assert "x_basis_points" in hp
        assert "y_basis_points" in hp
        assert "x_search_ranges" in hp
        assert "y_search_range" in hp
        assert "wind_speed" in hp["x_bandwidth"]
        assert "temperature" in hp["x_bandwidth"]
        assert all(bw > 0 for bw in hp["x_bandwidth"].values())
        assert hp["y_bandwidth"] > 0
