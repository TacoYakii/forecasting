"""E2E convergence test for all combiners.

Verifies that Horizontal (SLSQP), Vertical (SLSQP), and Angular (COBYLA)
optimizers converge and produce valid output across varying model counts.
This is an implementation-validation test, not a regular regression test.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    ParametricForecastResult,
    QuantileForecastResult,
)
from src.models.combining.angular import AngularCombiner
from src.models.combining.horizontal import HorizontalCombiner
from src.models.combining.vertical import VerticalCombiner
from src.utils.metrics import crps_quantile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H = 3


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


def _make_models(M, N=50):
    """Create M models with varying quality and observed data."""
    basis_index = pd.date_range("2024-01-01", periods=N, freq="h")
    results = [
        _make_parametric(
            basis_index, f"Model_{i}",
            loc_offset=i * 0.5,
            scale=0.5 + i * 0.3,
        )
        for i in range(M)
    ]
    rng = np.random.default_rng(99)
    observed = pd.DataFrame(
        rng.standard_normal((N, H)), index=basis_index
    )
    return results, observed


def _assert_valid_output(combined, combiner, N):
    """Check combined result is valid."""
    # Type and shape
    assert isinstance(combined, QuantileForecastResult)
    assert len(combined) == N
    assert combined.horizon == H

    ql = combiner.quantile_levels

    for h in range(1, H + 1):
        # Weights valid
        w = combiner.weights_[h]
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= -1e-8)

        # Finite values
        q_vals = np.column_stack(
            [combined.quantiles_data[q][:, h - 1] for q in ql]
        )
        assert np.all(np.isfinite(q_vals)), f"Non-finite at h={h}"

        # Monotone quantiles
        diffs = np.diff(q_vals, axis=1)
        assert np.all(diffs >= -1e-10), f"Non-monotone at h={h}"


def _compute_crps(combined, combiner, observed):
    """Compute mean CRPS across all horizons."""
    tau = combiner.quantile_levels
    obs_arr = observed.values
    crps_vals = []
    for h in range(1, H + 1):
        q = np.column_stack([
            combined.quantiles_data[q][:, h - 1]
            for q in tau
        ])
        crps_vals.append(
            crps_quantile(tau, q, obs_arr[:, h - 1], reduction="mean")
        )
    return np.mean(crps_vals)


# ---------------------------------------------------------------------------
# Test: Horizontal (SLSQP) convergence
# ---------------------------------------------------------------------------


class TestHorizontalConvergence:
    """Horizontal combiner with SLSQP optimization."""

    @pytest.mark.parametrize("M", [2, 5, 10])
    def test_slsqp_converges(self, M):
        results, observed = _make_models(M)
        combiner = HorizontalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        combiner.fit(results, observed)

        assert combiner.is_fitted_
        combined = combiner.combine(results)
        _assert_valid_output(combined, combiner, len(results[0]))

    @pytest.mark.parametrize("M", [2, 5, 10])
    def test_slsqp_crps_nonnegative(self, M):
        results, observed = _make_models(M)
        combiner = HorizontalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        combiner.fit(results, observed)
        combined = combiner.combine(results)
        crps = _compute_crps(combined, combiner, observed)

        assert np.isfinite(crps)
        assert crps >= 0


# ---------------------------------------------------------------------------
# Test: Vertical (simplex) convergence
# ---------------------------------------------------------------------------


class TestVerticalConvergence:
    """Vertical combiner with SLSQP optimization."""

    @pytest.mark.parametrize("M", [2, 5, 10])
    def test_simplex_converges(self, M):
        results, observed = _make_models(M)
        combiner = VerticalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        combiner.fit(results, observed)

        assert combiner.is_fitted_
        combined = combiner.combine(results)
        _assert_valid_output(combined, combiner, len(results[0]))

    @pytest.mark.parametrize("M", [2, 5, 10])
    def test_simplex_crps_nonnegative(self, M):
        results, observed = _make_models(M)
        combiner = VerticalCombiner(
            n_quantiles=19, fit_method="optimize"
        )
        combiner.fit(results, observed)
        combined = combiner.combine(results)
        crps = _compute_crps(combined, combiner, observed)

        assert np.isfinite(crps)
        assert crps >= 0


# ---------------------------------------------------------------------------
# Test: Angular (COBYLA) convergence
# ---------------------------------------------------------------------------


class TestAngularConvergence:
    """Angular combiner with COBYLA optimization."""

    @pytest.mark.parametrize("M", [2, 5, 10])
    def test_cobyla_converges(self, M):
        results, observed = _make_models(M)
        combiner = AngularCombiner(
            n_quantiles=19, fit_method="cobyla"
        )
        combiner.fit(results, observed)

        assert combiner.is_fitted_
        combined = combiner.combine(results)
        _assert_valid_output(combined, combiner, len(results[0]))

        # params_ stored for all horizons
        for h in range(1, H + 1):
            assert "degree" in combiner.params_[h]
            deg = combiner.params_[h]["degree"]
            assert 0 <= deg <= 90

    @pytest.mark.parametrize("M", [2, 5, 10])
    def test_cobyla_crps_nonnegative(self, M):
        results, observed = _make_models(M)
        combiner = AngularCombiner(
            n_quantiles=19, fit_method="cobyla"
        )
        combiner.fit(results, observed)
        combined = combiner.combine(results)
        crps = _compute_crps(combined, combiner, observed)

        assert np.isfinite(crps)
        assert crps >= 0

    @pytest.mark.parametrize("M", [2, 5, 10])
    def test_cobyla_beats_equal_weights(self, M):
        """COBYLA result should be at least as good as equal weights."""
        results, observed = _make_models(M)

        w_eq = np.ones(M) / M
        c_eq = AngularCombiner(
            n_quantiles=19, weights=w_eq, degree=45.0
        )
        c_opt = AngularCombiner(
            n_quantiles=19, fit_method="cobyla"
        )

        c_eq.fit(results, observed)
        c_opt.fit(results, observed)

        crps_eq = _compute_crps(
            c_eq.combine(results), c_eq, observed
        )
        crps_opt = _compute_crps(
            c_opt.combine(results), c_opt, observed
        )

        assert crps_opt <= crps_eq + 1e-6


# ---------------------------------------------------------------------------
# Test: Cross-combiner comparison
# ---------------------------------------------------------------------------


class TestCrossCombinerComparison:
    """Compare all three combiners on the same data."""

    @pytest.mark.parametrize("M", [2, 5])
    def test_all_produce_valid_output(self, M):
        """All combiners produce valid results on the same input."""
        results, observed = _make_models(M)

        combiners = {
            "horizontal": HorizontalCombiner(
                n_quantiles=19, fit_method="optimize"
            ),
            "vertical": VerticalCombiner(
                n_quantiles=19, fit_method="optimize"
            ),
            "angular": AngularCombiner(
                n_quantiles=19, fit_method="cobyla"
            ),
        }

        for name, combiner in combiners.items():
            combiner.fit(results, observed)
            combined = combiner.combine(results)
            _assert_valid_output(combined, combiner, len(results[0]))

    @pytest.mark.parametrize("M", [2, 5])
    def test_all_crps_in_same_ballpark(self, M):
        """All combiners should produce CRPS within reasonable range."""
        results, observed = _make_models(M)

        scores = {}
        for name, combiner in [
            ("horizontal", HorizontalCombiner(
                n_quantiles=19, fit_method="optimize"
            )),
            ("vertical", VerticalCombiner(
                n_quantiles=19, fit_method="optimize"
            )),
            ("angular", AngularCombiner(
                n_quantiles=19, fit_method="cobyla"
            )),
        ]:
            combiner.fit(results, observed)
            combined = combiner.combine(results)
            scores[name] = _compute_crps(combined, combiner, observed)

        # All should be finite and positive
        for name, score in scores.items():
            assert np.isfinite(score), f"{name} CRPS is not finite"
            assert score > 0, f"{name} CRPS is zero"

        # No combiner should be dramatically worse than others
        # (within 5x of the best)
        best = min(scores.values())
        for name, score in scores.items():
            assert score < best * 5, (
                f"{name} CRPS={score:.4f} is >5x worse "
                f"than best={best:.4f}"
            )
