"""Tests for DeterministicCombiner (MAE/MSE-optimal point ensemble)."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    DeterministicForecastResult,
    ParametricForecastResult,
    QuantileForecastResult,
)
from src.models.combining.deterministic import (
    DeterministicCombiner,
    _extract_point_array,
    _mae_weights_lp,
    _mse_weights_qp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, H = 40, 3


@pytest.fixture
def basis_index():
    return pd.date_range("2024-01-01", periods=N, freq="h")


def _make_parametric(
    basis_index, model_name, loc_offset=0.0, scale=0.5, seed=42
):
    """Build a ParametricForecastResult with Normal(loc+offset, scale)."""
    rng = np.random.default_rng(seed)
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


def _make_quantile_from_y(basis_index, model_name, y_target, noise_std=0.1):
    """Build a QuantileForecastResult whose median tracks y_target.

    Useful for testing perfect-forecast scenarios with skewed outputs.
    """
    n = len(basis_index)
    q_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    # Build (N, H) quantile arrays where level 0.5 == y_target
    # Other levels offset symmetrically
    offsets = {0.1: -2.0, 0.25: -1.0, 0.5: 0.0, 0.75: 1.0, 0.9: 2.0}
    quantiles_data = {
        q: y_target + offsets[q] * noise_std for q in q_levels
    }
    return QuantileForecastResult(
        quantiles_data=quantiles_data,
        basis_index=basis_index,
        model_name=model_name,
    )


@pytest.fixture
def three_models(basis_index):
    return [
        _make_parametric(basis_index, "m1", loc_offset=0.0, seed=1),
        _make_parametric(basis_index, "m2", loc_offset=0.1, seed=2),
        _make_parametric(basis_index, "m3", loc_offset=0.2, seed=3),
    ]


@pytest.fixture
def observed(basis_index):
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.standard_normal((N, H)),
        index=basis_index,
        columns=list(range(1, H + 1)),
    )


# ---------------------------------------------------------------------------
# Solver correctness
# ---------------------------------------------------------------------------


class TestSolvers:
    """Direct tests of _mae_weights_lp and _mse_weights_qp."""

    def test_mae_recovers_true_weights(self):
        """LP recovers near-true weights on linear synthetic data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 3))
        w_true = np.array([0.5, 0.3, 0.2])
        y = X @ w_true + 0.05 * rng.standard_normal(300)
        w = _mae_weights_lp(X, y)
        np.testing.assert_allclose(w, w_true, atol=0.05)
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_mse_recovers_true_weights(self):
        """QP recovers near-true weights on linear synthetic data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 3))
        w_true = np.array([0.5, 0.3, 0.2])
        y = X @ w_true + 0.05 * rng.standard_normal(300)
        w = _mse_weights_qp(X, y)
        np.testing.assert_allclose(w, w_true, atol=0.05)
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_mae_one_perfect_model(self):
        """LP puts all weight on a perfect predictor."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(200)
        # Model 0 is perfect, models 1 and 2 are noisy
        X = np.column_stack([
            y,
            y + rng.standard_normal(200),
            y + 2 * rng.standard_normal(200),
        ])
        w = _mae_weights_lp(X, y)
        assert w[0] > 0.99, f"Expected ~1 on perfect model, got {w}"
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_mse_one_perfect_model(self):
        """QP puts all weight on a perfect predictor."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(200)
        X = np.column_stack([
            y,
            y + rng.standard_normal(200),
            y + 2 * rng.standard_normal(200),
        ])
        w = _mse_weights_qp(X, y)
        assert w[0] > 0.99, f"Expected ~1 on perfect model, got {w}"
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_mae_weights_nonneg_and_sum_to_one(self):
        """LP output is always valid (non-negative, sums to 1)."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5))
        y = rng.standard_normal(100)
        w = _mae_weights_lp(X, y)
        assert np.all(w >= 0)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)

    def test_mse_weights_nonneg_and_sum_to_one(self):
        """QP output is always valid."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5))
        y = rng.standard_normal(100)
        w = _mse_weights_qp(X, y)
        assert np.all(w >= -1e-10)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Point extraction
# ---------------------------------------------------------------------------


class TestExtractPoint:
    """Tests for _extract_point_array."""

    def test_mean_shape(self, three_models):
        mu = _extract_point_array(three_models[0], point="mean")
        assert mu.shape == (N, H)

    def test_median_shape(self, three_models):
        mu = _extract_point_array(three_models[0], point="median")
        assert mu.shape == (N, H)

    def test_normal_mean_equals_median(self, basis_index):
        """For Normal distribution, mean == median."""
        r = _make_parametric(basis_index, "m", loc_offset=1.0, scale=0.5)
        mean = _extract_point_array(r, point="mean")
        median = _extract_point_array(r, point="median")
        np.testing.assert_allclose(mean, median, atol=1e-10)


# ---------------------------------------------------------------------------
# Core fit/combine contract
# ---------------------------------------------------------------------------


class TestFit:
    """Tests for DeterministicCombiner.fit()."""

    def test_learns_weights_per_horizon(self, three_models, observed):
        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        assert c.is_fitted_
        assert set(c.weights_.keys()) == set(range(1, H + 1))
        for h, w in c.weights_.items():
            assert w.shape == (3,)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)
            assert np.all(w >= 0)

    def test_train_scores_populated(self, three_models, observed):
        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        assert set(c.train_scores_.keys()) == set(range(1, H + 1))
        for _, score in c.train_scores_.items():
            assert score > 0
        # val_scores_ empty when val_ratio=0
        assert c.val_scores_ == {}

    def test_val_split(self, three_models, observed):
        c = DeterministicCombiner(n_jobs=1, val_ratio=0.25)
        c.fit(three_models, observed)
        assert set(c.val_scores_.keys()) == set(range(1, H + 1))

    def test_user_weights_bypass_learning(self, three_models, observed):
        user_w = np.array([0.6, 0.3, 0.1])
        c = DeterministicCombiner(n_jobs=1, weights=user_w)
        c.fit(three_models, observed)
        for h in range(1, H + 1):
            np.testing.assert_allclose(c.weights_[h], user_w)

    def test_raises_on_duplicate_model_names(self, basis_index, observed):
        r1 = _make_parametric(basis_index, "same", seed=1)
        r2 = _make_parametric(basis_index, "same", seed=2)
        c = DeterministicCombiner(n_jobs=1)
        with pytest.raises(ValueError, match="Duplicate model_name"):
            c.fit([r1, r2], observed)

    def test_raises_on_horizon_mismatch(self, basis_index, observed):
        rng = np.random.default_rng(0)
        r1 = _make_parametric(basis_index, "m1", seed=1)
        # Create a different-horizon result
        r2 = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": rng.standard_normal((N, H + 1)),
                "scale": np.full((N, H + 1), 0.5),
            },
            basis_index=basis_index,
            model_name="m2",
        )
        c = DeterministicCombiner(n_jobs=1)
        with pytest.raises(ValueError, match="Horizon mismatch"):
            c.fit([r1, r2], observed)

    def test_raises_on_fewer_than_two_models(
        self, three_models, observed
    ):
        c = DeterministicCombiner(n_jobs=1)
        with pytest.raises(ValueError, match="At least 2 models"):
            c.fit([three_models[0]], observed)

    def test_raises_on_bad_loss(self):
        with pytest.raises(ValueError, match="loss must be"):
            DeterministicCombiner(loss="crps")

    def test_raises_on_bad_point(self):
        with pytest.raises(ValueError, match="point must be"):
            DeterministicCombiner(point="mode")


class TestCombine:
    """Tests for DeterministicCombiner.combine()."""

    def test_output_type_and_shape(self, three_models, observed):
        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        out = c.combine(three_models)
        assert isinstance(out, DeterministicForecastResult)
        assert out.mu.shape == (N, H)

    def test_combine_before_fit_raises(self, three_models):
        c = DeterministicCombiner(n_jobs=1)
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            c.combine(three_models)

    def test_combine_respects_model_order(self, three_models, observed):
        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        # Reversed model order should raise
        with pytest.raises(ValueError, match="Model names/order mismatch"):
            c.combine(list(reversed(three_models)))

    def test_combine_applies_weights_correctly(
        self, basis_index, three_models, observed
    ):
        """Combined mu should equal weighted sum of individual means."""
        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        out = c.combine(three_models)

        # Manually compute for horizon 1
        expected = np.zeros(N)
        for i, r in enumerate(three_models):
            expected += c.weights_[1][i] * r.to_distribution(1).mean()
        np.testing.assert_allclose(out.mu[:, 0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Mean vs Median input behaviour
# ---------------------------------------------------------------------------


class TestPointChoice:
    """Tests checking mean/median input selection."""

    def test_symmetric_dist_gives_same_weights(
        self, basis_index, observed
    ):
        """For symmetric (Normal) dists, mean==median → same weights."""
        models = [
            _make_parametric(basis_index, "m1", seed=1),
            _make_parametric(basis_index, "m2", loc_offset=0.3, seed=2),
            _make_parametric(basis_index, "m3", loc_offset=-0.2, seed=3),
        ]
        c_mean = DeterministicCombiner(point="mean", n_jobs=1)
        c_median = DeterministicCombiner(point="median", n_jobs=1)
        c_mean.fit(models, observed)
        c_median.fit(models, observed)
        for h in range(1, H + 1):
            np.testing.assert_allclose(
                c_mean.weights_[h], c_median.weights_[h], atol=1e-6
            )

    def test_quantile_input_works(self, basis_index, observed):
        """Combiner handles QuantileForecastResult inputs."""
        rng = np.random.default_rng(0)
        y_target = rng.standard_normal((N, H))
        models = [
            _make_quantile_from_y(
                basis_index, f"q{i}", y_target + 0.05 * i, noise_std=0.1
            )
            for i in range(3)
        ]
        c = DeterministicCombiner(point="median", n_jobs=1)
        c.fit(models, observed)
        out = c.combine(models)
        assert out.mu.shape == (N, H)


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


class TestBiasCorrection:
    """Combiner should correct systematic biases via weighting."""

    def test_mae_reduces_error_below_best_individual(
        self, basis_index
    ):
        """Combined MAE ≤ min individual MAE when biases differ."""
        rng = np.random.default_rng(0)
        y_mat = rng.standard_normal((N, H))
        observed = pd.DataFrame(
            y_mat, index=basis_index, columns=list(range(1, H + 1))
        )
        # Three biased models with different bias directions
        m1 = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": y_mat + 0.3,
                "scale": np.full((N, H), 0.3),
            },
            basis_index=basis_index,
            model_name="pos_bias",
        )
        m2 = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": y_mat - 0.4,
                "scale": np.full((N, H), 0.3),
            },
            basis_index=basis_index,
            model_name="neg_bias",
        )
        m3 = ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": y_mat + 0.1,
                "scale": np.full((N, H), 0.3),
            },
            basis_index=basis_index,
            model_name="small_bias",
        )
        c = DeterministicCombiner(loss="mae", n_jobs=1)
        c.fit([m1, m2, m3], observed)
        out = c.combine([m1, m2, m3])

        # Individual MAEs at h=1
        indiv_mae = [
            float(np.mean(np.abs(
                r.to_distribution(1).mean() - y_mat[:, 0]
            )))
            for r in [m1, m2, m3]
        ]
        combined_mae = float(
            np.mean(np.abs(out.mu[:, 0] - y_mat[:, 0]))
        )
        assert combined_mae <= min(indiv_mae) + 1e-9


# ---------------------------------------------------------------------------
# Integration: to_nmape_frames compatibility
# ---------------------------------------------------------------------------


class TestAdapterCompat:
    """DeterministicForecastResult should work with to_nmape_frames."""

    def test_to_nmape_frames_roundtrip(
        self, three_models, observed, basis_index
    ):
        from src.utils.nMAPE import to_nmape_frames

        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        out = c.combine(three_models)

        frames = to_nmape_frames(out)
        assert set(frames.keys()) == set(range(1, H + 1))
        for h, df in frames.items():
            assert list(df.columns) == [
                "basis_time", "forecast_time", "mu"
            ]
            assert len(df) == N
            # mu should match combined mu
            np.testing.assert_allclose(
                df["mu"].to_numpy(), out.mu[:, h - 1]
            )

    def test_to_nmape_frames_median_equals_mean(
        self, three_models, observed
    ):
        """For DeterministicForecastResult, point='mean' == point='median'."""
        from src.utils.nMAPE import to_nmape_frames

        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        out = c.combine(three_models)

        f_mean = to_nmape_frames(out, point="mean")
        f_median = to_nmape_frames(out, point="median")
        for h in range(1, H + 1):
            np.testing.assert_allclose(
                f_mean[h]["mu"].to_numpy(),
                f_median[h]["mu"].to_numpy(),
            )


# ---------------------------------------------------------------------------
# Top-K selection + L2 regularization
# ---------------------------------------------------------------------------


class TestL2Regularization:
    """Tests for l2_to_uniform in _mse_weights_qp."""

    def test_l2_zero_matches_pure_mse(self):
        """l2=0 must yield identical weights to pure MSE QP."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((200, 4))
        y = rng.standard_normal(200)
        w0 = _mse_weights_qp(X, y, l2_to_uniform=0.0)
        w_pure = _mse_weights_qp(X, y)
        np.testing.assert_allclose(w0, w_pure, atol=1e-10)

    def test_l2_pulls_toward_uniform(self):
        """Large l2_to_uniform shrinks weights toward 1/M."""
        rng = np.random.default_rng(1)
        # Perfect first predictor; pure MSE would pick w[0] ≈ 1
        y = rng.standard_normal(300)
        X = np.column_stack([
            y,
            y + rng.standard_normal(300),
            y + 2 * rng.standard_normal(300),
            y + 3 * rng.standard_normal(300),
        ])
        uniform = np.full(4, 0.25)
        w_pure = _mse_weights_qp(X, y, l2_to_uniform=0.0)
        w_mild = _mse_weights_qp(X, y, l2_to_uniform=0.1)
        w_strong = _mse_weights_qp(X, y, l2_to_uniform=10.0)
        # Distance to uniform should decrease as lambda grows
        d_pure = np.linalg.norm(w_pure - uniform)
        d_mild = np.linalg.norm(w_mild - uniform)
        d_strong = np.linalg.norm(w_strong - uniform)
        assert d_pure > d_mild > d_strong
        # Very large lambda → essentially uniform
        w_huge = _mse_weights_qp(X, y, l2_to_uniform=1e6)
        np.testing.assert_allclose(w_huge, uniform, atol=1e-3)

    def test_l2_weights_remain_valid(self):
        """Weights remain non-negative and sum to 1 for any λ."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((150, 5))
        y = rng.standard_normal(150)
        for lam in [0.01, 0.5, 5.0, 100.0]:
            w = _mse_weights_qp(X, y, l2_to_uniform=lam)
            assert np.all(w >= -1e-10)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)


class TestTopKSelection:
    """Tests for top_k model subset selection in DeterministicCombiner."""

    def test_top_k_none_uses_all_models(self, three_models, observed):
        """top_k=None preserves original behavior with all M models."""
        c = DeterministicCombiner(n_jobs=1, top_k=None)
        c.fit(three_models, observed)
        assert c.top_k_indices_ == [0, 1, 2]
        for h, w in c.weights_.items():
            assert w.shape == (3,)
            assert np.all(w >= 0)
            np.testing.assert_allclose(w.sum(), 1.0)

    def test_top_k_selects_subset(self, basis_index):
        """With top_k < M, only the best K models get nonzero weight."""
        # Build 5 models with increasing noise → clear ranking
        rng = np.random.default_rng(11)
        y_true = rng.standard_normal((N, H))
        models = []
        for i in range(5):
            noise = (i + 1) * 0.3 * rng.standard_normal((N, H))
            models.append(
                ParametricForecastResult(
                    dist_name="normal",
                    params={"loc": y_true + noise, "scale": np.full((N, H), 0.5)},
                    basis_index=basis_index,
                    model_name=f"m{i}",
                )
            )
        obs = pd.DataFrame(
            y_true, index=basis_index, columns=list(range(1, H + 1))
        )
        c = DeterministicCombiner(n_jobs=1, top_k=2)
        c.fit(models, obs)
        assert len(c.top_k_indices_) == 2
        # Excluded models must have weight 0
        for h, w in c.weights_.items():
            assert w.shape == (5,)
            excluded = [i for i in range(5) if i not in c.top_k_indices_]
            for j in excluded:
                assert w[j] == 0.0
            np.testing.assert_allclose(w.sum(), 1.0)
        # The two best (lowest-noise) models are m0 and m1
        assert 0 in c.top_k_indices_
        assert 1 in c.top_k_indices_

    def test_top_k_greater_than_m_uses_all(self, three_models, observed):
        """top_k >= M falls back to all models."""
        c = DeterministicCombiner(n_jobs=1, top_k=10)
        c.fit(three_models, observed)
        assert c.top_k_indices_ == [0, 1, 2]

    def test_model_scores_populated(self, three_models, observed):
        """model_scores_ dict is populated per model."""
        c = DeterministicCombiner(n_jobs=1, top_k=2)
        c.fit(three_models, observed)
        assert set(c.model_scores_.keys()) == {"m1", "m2", "m3"}
        for v in c.model_scores_.values():
            assert v >= 0.0

    def test_combine_uses_learned_subset(self, basis_index):
        """combine() applies zero weights to excluded models automatically."""
        rng = np.random.default_rng(22)
        y_true = rng.standard_normal((N, H))
        models = [
            ParametricForecastResult(
                dist_name="normal",
                params={
                    "loc": y_true + (i + 1) * 0.2 * rng.standard_normal((N, H)),
                    "scale": np.full((N, H), 0.5),
                },
                basis_index=basis_index,
                model_name=f"m{i}",
            )
            for i in range(4)
        ]
        obs = pd.DataFrame(y_true, index=basis_index, columns=list(range(1, H + 1)))
        c = DeterministicCombiner(n_jobs=1, top_k=2)
        c.fit(models, obs)
        out = c.combine(models)
        assert out.mu.shape == (N, H)
        assert np.all(np.isfinite(out.mu))


class TestDeterministicCombinerValidation:
    """Validation for new parameters."""

    def test_top_k_too_small_raises(self):
        with pytest.raises(ValueError, match="top_k must be >= 2"):
            DeterministicCombiner(top_k=1)

    def test_l2_negative_raises(self):
        with pytest.raises(ValueError, match="l2_to_uniform must be >= 0"):
            DeterministicCombiner(l2_to_uniform=-0.5)

    def test_l2_with_mae_raises(self):
        with pytest.raises(
            ValueError, match="l2_to_uniform is only supported with loss='mse'"
        ):
            DeterministicCombiner(loss="mae", l2_to_uniform=1.0)

    def test_l2_zero_with_mae_ok(self):
        """l2_to_uniform=0 with MAE loss is allowed."""
        c = DeterministicCombiner(loss="mae", l2_to_uniform=0.0)
        assert c.l2_to_uniform == 0.0


class TestLOMOCrossValidation:
    """Tests for leave-one-month-out CV to select l2_to_uniform."""

    def _multi_month_index(self, n_per_month=20, n_months=4):
        """Build a basis_index spanning n_months distinct calendar months."""
        dates = []
        for m in range(n_months):
            start = pd.Timestamp(f"2024-{m + 1:02d}-01")
            dates.extend(
                pd.date_range(start, periods=n_per_month, freq="h")
            )
        return pd.DatetimeIndex(dates)

    def test_cv_grid_selects_from_candidates(self):
        """CV must select a λ from the provided grid."""
        basis = self._multi_month_index(n_per_month=30, n_months=3)
        n = len(basis)
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal((n, H))
        models = [
            ParametricForecastResult(
                dist_name="normal",
                params={
                    "loc": y_true + (i + 1) * 0.3 * rng.standard_normal((n, H)),
                    "scale": np.full((n, H), 0.5),
                },
                basis_index=basis,
                model_name=f"m{i}",
            )
            for i in range(4)
        ]
        obs = pd.DataFrame(y_true, index=basis, columns=list(range(1, H + 1)))
        grid = (0.0, 0.5, 2.0)
        c = DeterministicCombiner(
            loss="mse", top_k=3, l2_cv_grid=grid, n_jobs=1
        )
        c.fit(models, obs)
        assert c.l2_to_uniform_ in grid
        assert set(c.l2_cv_scores_.keys()) == set(grid)
        # All CV scores should be positive MAE
        for lam, score in c.l2_cv_scores_.items():
            assert score > 0

    def test_cv_grid_overrides_l2_to_uniform(self):
        """Selected l2 may differ from user's l2_to_uniform."""
        basis = self._multi_month_index(n_per_month=30, n_months=3)
        n = len(basis)
        rng = np.random.default_rng(11)
        y_true = rng.standard_normal((n, H))
        models = [
            ParametricForecastResult(
                dist_name="normal",
                params={
                    "loc": y_true + (i + 1) * 0.2 * rng.standard_normal((n, H)),
                    "scale": np.full((n, H), 0.5),
                },
                basis_index=basis,
                model_name=f"m{i}",
            )
            for i in range(4)
        ]
        obs = pd.DataFrame(y_true, index=basis, columns=list(range(1, H + 1)))
        c = DeterministicCombiner(
            loss="mse",
            top_k=3,
            l2_to_uniform=999.0,  # would be selected without CV
            l2_cv_grid=(0.0, 0.1, 1.0),
            n_jobs=1,
        )
        c.fit(models, obs)
        # Effective λ is CV-selected, not 999.0
        assert c.l2_to_uniform_ in (0.0, 0.1, 1.0)
        assert c.l2_to_uniform_ != 999.0

    def test_cv_single_month_raises(self, three_models, observed):
        """LOMO CV requires >= 2 distinct months."""
        c = DeterministicCombiner(
            loss="mse", top_k=2, l2_cv_grid=(0.0, 1.0), n_jobs=1
        )
        with pytest.raises(ValueError, match="LOMO CV requires >=2 distinct months"):
            c.fit(three_models, observed)

    def test_cv_grid_with_mae_raises(self):
        with pytest.raises(ValueError, match="l2_cv_grid is only supported"):
            DeterministicCombiner(loss="mae", l2_cv_grid=(0.0, 1.0))

    def test_cv_grid_empty_raises(self):
        with pytest.raises(ValueError, match="l2_cv_grid must be non-empty"):
            DeterministicCombiner(loss="mse", l2_cv_grid=())

    def test_cv_grid_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DeterministicCombiner(loss="mse", l2_cv_grid=(-0.1, 0.5))


class TestSaveLoad:
    """Tests for DeterministicCombiner.save() / load() persistence."""

    def test_roundtrip_mae(self, three_models, observed, tmp_path):
        """Save → load roundtrip preserves weights and config (MAE)."""
        c = DeterministicCombiner(loss="mae", n_jobs=1)
        c.fit(three_models, observed)
        save_dir = c.save(tmp_path / "det_mae")

        loaded = DeterministicCombiner.load(save_dir)
        assert loaded.is_fitted_
        assert loaded.point == c.point
        assert loaded.loss == c.loss
        assert loaded.model_names_ == c.model_names_
        assert loaded._horizon == c._horizon
        assert loaded._n_models == c._n_models
        for h in range(1, H + 1):
            np.testing.assert_array_equal(
                loaded.weights_[h], c.weights_[h]
            )
            assert loaded.train_scores_[h] == c.train_scores_[h]

    def test_roundtrip_mse_with_topk(self, basis_index, tmp_path):
        """Save → load preserves top_k_indices and l2 config."""
        rng = np.random.default_rng(11)
        y_true = rng.standard_normal((N, H))
        models = [
            ParametricForecastResult(
                dist_name="normal",
                params={
                    "loc": y_true + (i + 1) * 0.3 * rng.standard_normal((N, H)),
                    "scale": np.full((N, H), 0.5),
                },
                basis_index=basis_index,
                model_name=f"m{i}",
            )
            for i in range(5)
        ]
        obs = pd.DataFrame(
            y_true, index=basis_index, columns=list(range(1, H + 1))
        )
        c = DeterministicCombiner(
            loss="mse", top_k=3, l2_to_uniform=0.5, n_jobs=1
        )
        c.fit(models, obs)
        save_dir = c.save(tmp_path / "det_mse_topk")

        loaded = DeterministicCombiner.load(save_dir)
        assert loaded.top_k == 3
        assert loaded.top_k_indices_ == c.top_k_indices_
        assert loaded.l2_to_uniform_ == c.l2_to_uniform_
        assert loaded.model_scores_ == c.model_scores_
        for h in range(1, H + 1):
            np.testing.assert_array_equal(
                loaded.weights_[h], c.weights_[h]
            )

    def test_loaded_combine_matches_original(
        self, three_models, observed, tmp_path
    ):
        """Loaded combiner produces identical combine() output."""
        c = DeterministicCombiner(n_jobs=1)
        c.fit(three_models, observed)
        out_orig = c.combine(three_models)

        loaded = DeterministicCombiner.load(c.save(tmp_path / "det"))
        out_loaded = loaded.combine(three_models)
        np.testing.assert_array_equal(out_orig.mu, out_loaded.mu)

    def test_save_before_fit_raises(self, tmp_path):
        """save() without fit() raises RuntimeError."""
        c = DeterministicCombiner()
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            c.save(tmp_path / "unfitted")

    def test_load_missing_dir_raises(self, tmp_path):
        """Loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DeterministicCombiner.load(tmp_path / "nonexistent")


class TestL2CombinerIntegration:
    """End-to-end: top_k + l2_to_uniform together."""

    def test_recommended_config_runs(self, basis_index):
        """top_k=6, l2_to_uniform=1.0 on 8 models — smoke test."""
        rng = np.random.default_rng(33)
        y_true = rng.standard_normal((N, H))
        models = [
            ParametricForecastResult(
                dist_name="normal",
                params={
                    "loc": y_true + (i + 1) * 0.1 * rng.standard_normal((N, H)),
                    "scale": np.full((N, H), 0.5),
                },
                basis_index=basis_index,
                model_name=f"m{i}",
            )
            for i in range(8)
        ]
        obs = pd.DataFrame(y_true, index=basis_index, columns=list(range(1, H + 1)))
        c = DeterministicCombiner(
            loss="mse", top_k=6, l2_to_uniform=1.0, n_jobs=1
        )
        c.fit(models, obs)
        assert len(c.top_k_indices_) == 6
        out = c.combine(models)
        assert out.mu.shape == (N, H)
        # Weights of 6 kept models should be closer to 1/6 than pure MSE
        for h, w in c.weights_.items():
            kept = w[c.top_k_indices_]
            # Pulled toward uniform 1/6 ≈ 0.167
            assert np.all(kept > 0.01)  # none fully zeroed out
