"""Tests for GridDistribution (histogram-based non-parametric distribution).

Covers:
- Constructor validation (grid, prob, edge cases)
- Moments (mean, std with within-bin correction)
- CDF / PDF / PPF correctness and round-trips
- Sampling statistics
- grid_crps closed-form integration (interior, boundary, tail, NaN)
- Interval, to_dataframe, to_grid_dataframe, get_dist_info
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics.crps import grid_crps

from src.core.forecast_distribution import GridDistribution


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def uniform_gd():
    """Uniform distribution on [0, 100] with 51 bins."""
    grid = np.linspace(0, 100, 51)
    prob = np.ones((5, 51)) / 51
    return GridDistribution(pd.RangeIndex(5), grid, prob)


@pytest.fixture
def point_mass_gd():
    """Near-point-mass at grid center (bin index 25 out of 51)."""
    grid = np.linspace(0, 100, 51)
    prob = np.zeros((3, 51))
    prob[:, 25] = 1.0  # all mass at grid[25] = 50
    return GridDistribution(pd.RangeIndex(3), grid, prob)


@pytest.fixture
def asymmetric_gd():
    """Linearly increasing probability: p_i ∝ i."""
    grid = np.linspace(0, 10, 21)
    weights = np.arange(1, 22, dtype=float)
    prob_row = weights / weights.sum()
    prob = np.tile(prob_row, (4, 1))
    return GridDistribution(pd.RangeIndex(4), grid, prob)


# =====================================================================
# Constructor validation
# =====================================================================

class TestConstructorValidation:
    """GridDistribution constructor edge cases and error handling."""

    def test_valid_construction(self, uniform_gd):
        """Basic construction succeeds with valid inputs."""
        assert len(uniform_gd) == 5
        assert uniform_gd.grid.shape == (51,)
        assert uniform_gd.prob.shape == (5, 51)

    def test_grid_not_1d(self):
        """2-D grid raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            GridDistribution(
                pd.RangeIndex(2),
                np.ones((2, 3)),
                np.ones((2, 3)) / 3,
            )

    def test_grid_too_short(self):
        """Single-point grid raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            GridDistribution(
                pd.RangeIndex(1),
                np.array([5.0]),
                np.array([[1.0]]),
            )

    def test_grid_not_increasing(self):
        """Non-monotone grid raises ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            GridDistribution(
                pd.RangeIndex(1),
                np.array([3.0, 1.0, 2.0]),
                np.ones((1, 3)) / 3,
            )

    def test_grid_not_equally_spaced(self):
        """Non-uniform grid raises ValueError."""
        with pytest.raises(ValueError, match="equally spaced"):
            GridDistribution(
                pd.RangeIndex(1),
                np.array([0.0, 1.0, 3.0]),
                np.ones((1, 3)) / 3,
            )

    def test_grid_with_nan(self):
        """NaN in grid raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            GridDistribution(
                pd.RangeIndex(1),
                np.array([0.0, np.nan, 2.0]),
                np.ones((1, 3)) / 3,
            )

    def test_grid_with_inf(self):
        """Inf in grid raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            GridDistribution(
                pd.RangeIndex(1),
                np.array([0.0, 1.0, np.inf]),
                np.ones((1, 3)) / 3,
            )

    def test_prob_negative(self):
        """Negative probability raises ValueError."""
        grid = np.linspace(0, 1, 3)
        with pytest.raises(ValueError, match="negative"):
            GridDistribution(
                pd.RangeIndex(1),
                grid,
                np.array([[0.5, -0.1, 0.6]]),
            )

    def test_prob_nan(self):
        """NaN in prob raises ValueError."""
        grid = np.linspace(0, 1, 3)
        with pytest.raises(ValueError, match="non-finite"):
            GridDistribution(
                pd.RangeIndex(1),
                grid,
                np.array([[0.5, np.nan, 0.5]]),
            )

    def test_prob_shape_mismatch(self):
        """Prob shape inconsistent with index/grid raises ValueError."""
        grid = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="inconsistent"):
            GridDistribution(
                pd.RangeIndex(3),
                grid,
                np.ones((2, 5)) / 5,  # T=2 != len(index)=3
            )

    def test_prob_row_sum_large_deviation(self):
        """Row sum deviating >1e-4 raises ValueError."""
        grid = np.linspace(0, 1, 3)
        with pytest.raises(ValueError, match="row sums"):
            GridDistribution(
                pd.RangeIndex(1),
                grid,
                np.array([[0.5, 0.5, 0.5]]),  # sum = 1.5
            )

    def test_prob_row_sum_small_deviation_auto_normalized(self):
        """Small float deviations (<=1e-4) are auto-renormalized."""
        grid = np.linspace(0, 1, 3)
        prob = np.array([[0.3333, 0.3333, 0.3334]])  # sum ≈ 1.0000
        gd = GridDistribution(pd.RangeIndex(1), grid, prob)
        np.testing.assert_allclose(gd.prob.sum(axis=1), 1.0)

    def test_two_point_grid(self):
        """Minimum viable grid (2 points) works correctly."""
        grid = np.array([0.0, 1.0])
        prob = np.array([[0.3, 0.7]])
        gd = GridDistribution(pd.RangeIndex(1), grid, prob)
        assert gd._bin_width == 1.0
        assert len(gd._edges) == 3

    def test_base_idx_optional(self):
        """base_idx is optional and defaults to None."""
        grid = np.linspace(0, 1, 5)
        gd = GridDistribution(pd.RangeIndex(2), grid, np.ones((2, 5)) / 5)
        assert gd.base_idx is None

    def test_base_idx_passed(self):
        """base_idx is stored when provided."""
        grid = np.linspace(0, 1, 5)
        bidx = pd.date_range("2023-01-01", periods=2, freq="h")
        gd = GridDistribution(
            pd.RangeIndex(2), grid, np.ones((2, 5)) / 5, base_idx=bidx
        )
        assert gd.base_idx is not None


# =====================================================================
# Moments
# =====================================================================

class TestMoments:
    """Mean and std computation."""

    def test_uniform_mean(self, uniform_gd):
        """Uniform dist on [0,100]: mean ≈ 50."""
        np.testing.assert_allclose(uniform_gd.mean(), 50.0, atol=0.1)

    def test_uniform_std(self, uniform_gd):
        """Uniform dist on [0,100]: std ≈ 100/√12 ≈ 28.87 (with bin correction)."""
        expected_var = (100.0 ** 2) / 12  # continuous uniform variance
        # Histogram approximation is close but not exact
        assert np.all(uniform_gd.std() > 25.0)
        assert np.all(uniform_gd.std() < 35.0)

    def test_point_mass_mean(self, point_mass_gd):
        """Point mass at 50: mean = 50."""
        np.testing.assert_allclose(point_mass_gd.mean(), 50.0)

    def test_point_mass_std(self, point_mass_gd):
        """Point mass at 50: std ≈ bin_width/√12 (within-bin correction only)."""
        bw = point_mass_gd._bin_width
        expected = bw / np.sqrt(12)
        np.testing.assert_allclose(point_mass_gd.std(), expected, atol=0.01)

    def test_mean_shape(self, uniform_gd):
        """Mean returns shape (T,)."""
        assert uniform_gd.mean().shape == (5,)

    def test_std_nonnegative(self, asymmetric_gd):
        """Std is always non-negative."""
        assert np.all(asymmetric_gd.std() >= 0)


# =====================================================================
# CDF / PDF / PPF
# =====================================================================

class TestCdfPdfPpf:
    """CDF, PDF, PPF correctness."""

    def test_cdf_boundaries(self, uniform_gd):
        """CDF at left edge ≈ 0, at right edge ≈ 1."""
        left = uniform_gd._edges[0]
        right = uniform_gd._edges[-1]
        np.testing.assert_allclose(uniform_gd.cdf(left), 0.0, atol=1e-10)
        np.testing.assert_allclose(uniform_gd.cdf(right), 1.0, atol=1e-10)

    def test_cdf_monotone(self, asymmetric_gd):
        """CDF is monotonically non-decreasing."""
        xs = np.linspace(
            asymmetric_gd._edges[0], asymmetric_gd._edges[-1], 200
        )
        T = len(asymmetric_gd)
        for t in range(T):
            # Broadcast scalar x to all T, pick row t
            vals = np.array([asymmetric_gd.cdf(np.full(T, x))[t] for x in xs])
            assert np.all(np.diff(vals) >= -1e-12)

    def test_cdf_outside_support(self, uniform_gd):
        """CDF returns 0 below support, 1 above support."""
        np.testing.assert_allclose(uniform_gd.cdf(-100.0), 0.0, atol=1e-10)
        np.testing.assert_allclose(uniform_gd.cdf(200.0), 1.0, atol=1e-10)

    def test_uniform_cdf_midpoint(self, uniform_gd):
        """Uniform: CDF(50) ≈ 0.5."""
        np.testing.assert_allclose(uniform_gd.cdf(50.0), 0.5, atol=0.02)

    def test_pdf_inside_support(self, uniform_gd):
        """Uniform: pdf ≈ 1/range inside support."""
        bw = uniform_gd._bin_width
        expected_density = (1.0 / 51) / bw
        vals = uniform_gd.pdf(50.0)
        np.testing.assert_allclose(vals, expected_density, rtol=0.01)

    def test_pdf_outside_support(self, uniform_gd):
        """PDF is 0 outside support."""
        assert np.all(uniform_gd.pdf(-10.0) == 0.0)
        assert np.all(uniform_gd.pdf(200.0) == 0.0)

    def test_pdf_nan(self, uniform_gd):
        """PDF returns 0 for NaN input."""
        vals = uniform_gd.pdf(np.full(5, np.nan))
        np.testing.assert_array_equal(vals, 0.0)

    def test_ppf_boundaries(self, uniform_gd):
        """PPF(0) ≈ left edge, PPF(1) ≈ right edge."""
        left = uniform_gd._edges[0]
        right = uniform_gd._edges[-1]
        np.testing.assert_allclose(uniform_gd.ppf(0.0), left, atol=1e-10)
        np.testing.assert_allclose(uniform_gd.ppf(1.0), right, atol=1e-10)

    def test_ppf_median_uniform(self, uniform_gd):
        """Uniform: median ≈ 50."""
        np.testing.assert_allclose(uniform_gd.ppf(0.5), 50.0, atol=1.5)

    def test_ppf_vector(self, uniform_gd):
        """PPF with vector q returns (T, Q)."""
        result = uniform_gd.ppf([0.1, 0.5, 0.9])
        assert result.shape == (5, 3)

    def test_cdf_ppf_roundtrip(self, asymmetric_gd):
        """CDF(PPF(q)) ≈ q for q in (0, 1)."""
        T = len(asymmetric_gd)
        qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        ppf_vals = asymmetric_gd.ppf(qs)  # (T, Q)
        for t in range(T):
            for j, q in enumerate(qs):
                x_val = np.full(T, ppf_vals[t, j])
                cdf_val = asymmetric_gd.cdf(x_val)[t]
                np.testing.assert_allclose(cdf_val, q, atol=0.02)


# =====================================================================
# Sampling
# =====================================================================

class TestSampling:
    """Sample generation."""

    def test_sample_shape(self, uniform_gd):
        """Sample returns (T, n)."""
        s = uniform_gd.sample(500, seed=0)
        assert s.shape == (5, 500)

    def test_sample_mean_convergence(self, uniform_gd):
        """Sample mean converges to distribution mean."""
        s = uniform_gd.sample(50000, seed=42)
        np.testing.assert_allclose(s.mean(axis=1), uniform_gd.mean(), atol=1.0)

    def test_sample_reproducibility(self, uniform_gd):
        """Same seed → same samples."""
        s1 = uniform_gd.sample(100, seed=123)
        s2 = uniform_gd.sample(100, seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_sample_different_seeds(self, uniform_gd):
        """Different seeds → different samples."""
        s1 = uniform_gd.sample(100, seed=1)
        s2 = uniform_gd.sample(100, seed=2)
        assert not np.array_equal(s1, s2)

    def test_point_mass_sample_range(self, point_mass_gd):
        """Point mass samples cluster within one bin of the center."""
        s = point_mass_gd.sample(1000, seed=0)
        bw = point_mass_gd._bin_width
        assert np.all(s >= 50.0 - bw / 2 - 1e-10)
        assert np.all(s <= 50.0 + bw / 2 + 1e-10)


# =====================================================================
# CRPS
# =====================================================================

class TestCRPS:
    """Closed-form CRPS integration."""

    def test_crps_perfect_forecast(self, point_mass_gd):
        """CRPS ≈ 0 when observation matches the point mass (within bin width)."""
        y = np.full(3, 50.0)
        result = grid_crps(point_mass_gd, y)
        # Not exactly 0 because histogram has bin width, but should be small
        assert np.all(result < point_mass_gd._bin_width)

    def test_crps_nonnegative(self, uniform_gd):
        """CRPS is always non-negative."""
        y = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = grid_crps(uniform_gd, y)
        assert np.all(result >= 0)

    def test_crps_nan_propagation(self, uniform_gd):
        """NaN observation → NaN CRPS."""
        y = np.array([50.0, np.nan, 50.0, np.nan, 50.0])
        result = grid_crps(uniform_gd, y)
        assert np.isnan(result[1])
        assert np.isnan(result[3])
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        assert np.isfinite(result[4])

    def test_crps_observation_below_support(self, uniform_gd):
        """Observation below grid: CRPS includes left tail contribution."""
        y_inside = np.full(5, 0.0)
        y_below = np.full(5, -50.0)
        crps_inside = grid_crps(uniform_gd, y_inside)
        crps_below = grid_crps(uniform_gd, y_below)
        # Below-support CRPS should be larger
        assert np.all(crps_below > crps_inside)

    def test_crps_observation_above_support(self, uniform_gd):
        """Observation above grid: CRPS includes right tail contribution."""
        y_inside = np.full(5, 100.0)
        y_above = np.full(5, 200.0)
        crps_inside = grid_crps(uniform_gd, y_inside)
        crps_above = grid_crps(uniform_gd, y_above)
        assert np.all(crps_above > crps_inside)

    def test_crps_tail_linear(self, uniform_gd):
        """Tail contribution is linear in distance from support."""
        y1 = np.full(5, -10.0)
        y2 = np.full(5, -20.0)
        crps1 = grid_crps(uniform_gd, y1)
        crps2 = grid_crps(uniform_gd, y2)
        # Difference should be close to 10 (tail adds exactly the distance)
        np.testing.assert_allclose(crps2 - crps1, 10.0, atol=0.1)

    def test_crps_symmetry(self):
        """Symmetric distribution: CRPS(mean+d) ≈ CRPS(mean-d)."""
        grid = np.linspace(-10, 10, 41)
        # symmetric triangular-ish
        weights = 20.0 - np.abs(np.arange(41) - 20.0)
        prob = (weights / weights.sum())[np.newaxis, :]
        gd = GridDistribution(pd.RangeIndex(1), grid, prob)
        crps_pos = grid_crps(gd, np.array([3.0]))
        crps_neg = grid_crps(gd, np.array([-3.0]))
        np.testing.assert_allclose(crps_pos, crps_neg, rtol=0.05)

    def test_crps_vs_sample_based(self, asymmetric_gd):
        """Grid-exact CRPS should be close to sample-based numerical CRPS."""
        from src.utils.metrics.crps import crps_numerical
        y = asymmetric_gd.mean()  # observe at the mean
        exact = grid_crps(asymmetric_gd, y)
        samples = asymmetric_gd.sample(10000, seed=0)
        numerical = np.array([
            crps_numerical(y[i:i + 1], samples[i:i + 1])
            for i in range(len(asymmetric_gd))
        ])
        np.testing.assert_allclose(exact, numerical, rtol=0.1)

    def test_crps_observation_at_bin_edge(self, uniform_gd):
        """Observation exactly at a bin edge doesn't crash or produce NaN."""
        edges = uniform_gd._edges
        y = np.full(5, edges[10])
        result = grid_crps(uniform_gd, y)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


# =====================================================================
# Interval, DataFrame, info
# =====================================================================

class TestMiscMethods:
    """Interval, to_dataframe, to_grid_dataframe, get_dist_info."""

    def test_interval_coverage(self, uniform_gd):
        """90% interval: lower < upper, roughly spans 90% of support."""
        lower, upper = uniform_gd.interval(0.9)
        assert np.all(lower < upper)
        assert lower.shape == (5,)

    def test_interval_100pct(self, uniform_gd):
        """100% interval spans full support."""
        lower, upper = uniform_gd.interval(1.0)
        np.testing.assert_allclose(lower, uniform_gd._edges[0], atol=1e-6)
        np.testing.assert_allclose(upper, uniform_gd._edges[-1], atol=1e-6)

    def test_to_dataframe(self, uniform_gd):
        """to_dataframe returns {mu, std} summary."""
        df = uniform_gd.to_dataframe()
        assert set(df.columns) == {"mu", "std"}
        assert len(df) == 5

    def test_to_dataframe_with_base_idx(self):
        """to_dataframe includes basis_time when base_idx is set."""
        grid = np.linspace(0, 1, 5)
        bidx = pd.date_range("2023-01-01", periods=2, freq="h")
        gd = GridDistribution(
            pd.RangeIndex(2), grid, np.ones((2, 5)) / 5, base_idx=bidx
        )
        df = gd.to_dataframe()
        assert "basis_time" in df.columns

    def test_to_grid_dataframe(self, uniform_gd):
        """to_grid_dataframe returns full (T, G) probability matrix."""
        df = uniform_gd.to_grid_dataframe()
        assert df.shape == (5, 51)
        np.testing.assert_allclose(df.values.sum(axis=1), 1.0)

    def test_get_dist_info(self, uniform_gd):
        """get_dist_info returns correct metadata."""
        info = uniform_gd.get_dist_info()
        assert info["dist_name"] == "grid"
        assert info["G"] == 51
        assert info["T"] == 5
        assert "bin_width" in info
        assert "range" in info

    def test_repr(self, uniform_gd):
        """repr is informative."""
        r = repr(uniform_gd)
        assert "GridDistribution" in r
        assert "T=5" in r
        assert "G=51" in r

    def test_len(self, uniform_gd):
        """len returns T."""
        assert len(uniform_gd) == 5


# =====================================================================
# Cross-distribution CRPS consistency
# =====================================================================

class TestCRPSCrossDistribution:
    """Same distribution represented as Grid/Parametric/Sample/Quantile.

    All four representations should yield similar CRPS values,
    verifying that grid_crps and crps_numerical are consistent.
    """

    def test_normal_crps_all_representations(self):
        """N(50, 10): Grid vs Parametric vs Sample vs Quantile CRPS."""
        from scipy import stats
        from src.core.forecast_distribution import (
            GridDistribution,
            ParametricDistribution,
            SampleDistribution,
            QuantileDistribution,
        )
        from src.utils.metrics.crps import crps_numerical, grid_crps

        T = 20
        mu, sigma = 50.0, 10.0
        rng = np.random.default_rng(42)
        y_obs = rng.normal(mu, sigma, T)
        idx = pd.RangeIndex(T)

        # --- GridDistribution ---
        G = 501
        grid = np.linspace(0, 100, G)
        rv = stats.norm(loc=mu, scale=sigma)
        raw_prob = rv.pdf(grid)
        prob = np.tile(raw_prob / raw_prob.sum(), (T, 1))
        gd = GridDistribution(idx, grid, prob)
        crps_grid = grid_crps(gd, y_obs)

        # --- ParametricDistribution → pseudo-sample CRPS ---
        pd_ = ParametricDistribution(
            dist_name="normal",
            params={"loc": np.full(T, mu), "scale": np.full(T, sigma)},
            index=idx,
        )
        tau = np.linspace(0, 1, 201)[1:-1]
        parametric_samples = pd_.ppf(tau)  # (T, 199)
        crps_parametric = np.array([
            crps_numerical(y_obs[i:i+1], parametric_samples[i:i+1])
            for i in range(T)
        ])

        # --- SampleDistribution ---
        raw_samples = rng.normal(mu, sigma, (T, 10000))
        sd = SampleDistribution(idx, raw_samples)
        sample_pseudo = sd.ppf(tau)  # (T, 199)
        crps_sample = np.array([
            crps_numerical(y_obs[i:i+1], sample_pseudo[i:i+1])
            for i in range(T)
        ])

        # --- QuantileDistribution ---
        levels = np.linspace(0, 1, 101)[1:-1]
        qvals = rv.ppf(levels)[np.newaxis, :].repeat(T, axis=0)
        qd = QuantileDistribution(idx, levels, qvals)
        quantile_pseudo = qd.ppf(tau)  # (T, 199)
        crps_quantile = np.array([
            crps_numerical(y_obs[i:i+1], quantile_pseudo[i:i+1])
            for i in range(T)
        ])

        # All four should agree within 10%
        mean_grid = crps_grid.mean()
        mean_parametric = crps_parametric.mean()
        mean_sample = crps_sample.mean()
        mean_quantile = crps_quantile.mean()

        np.testing.assert_allclose(mean_parametric, mean_grid, rtol=0.10)
        np.testing.assert_allclose(mean_sample, mean_grid, rtol=0.10)
        np.testing.assert_allclose(mean_quantile, mean_grid, rtol=0.10)
