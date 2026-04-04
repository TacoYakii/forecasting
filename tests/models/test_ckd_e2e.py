"""End-to-end integration test for CKD adaptive bandwidth pipeline.

Tests the full pipeline with timing:
    1. Synthetic power curve data → train / validation split
    2. Training data → automatic search space determination
    3. Optuna hyperparameter optimization (bandwidth + decay)
    4. CKDRunner per-horizon parallel estimation
    5. CRPS evaluation on held-out validation data

Reports timing for each stage to understand scaling characteristics.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.core.ckd_runner import CKDRunner
from src.core.forecast_results import (
    GridForecastResult,
    ParametricForecastResult,
)
from src.models.conditional_kernel_density import (
    CKDConfig,
    ConditionalKernelDensity,
)
from src.trainers.ckd import CKDOptunaConfig, CKDOptunaTrainer
from src.utils.metrics.crps import grid_crps


# =====================================================================
# Helpers
# =====================================================================

def _make_power_curve_data(n: int, seed: int = 0):
    """Simulate wind speed → power with a sigmoid power curve.

    Mimics real wind farm behaviour:
    - cut-in ≈ 3 m/s, rated ≈ 12 m/s, cut-out ≈ 25 m/s
    - Gaussian noise proportional to wind speed

    Returns:
        x: shape (n, 1), wind speed
        y: shape (n,), power output (MW)
    """
    rng = np.random.default_rng(seed)
    wind_speed = rng.uniform(0, 25, n)
    # Sigmoid power curve
    rated_power = 3.0  # MW
    power = rated_power / (1 + np.exp(-0.8 * (wind_speed - 8)))
    # Below cut-in
    power[wind_speed < 3] = 0.0
    # Above cut-out
    power[wind_speed > 23] = 0.0
    # Add noise
    noise = rng.normal(0, 0.1 * power + 0.05, n)
    power = np.clip(power + noise, 0, rated_power)
    x = wind_speed[:, None]
    return x, power


def _make_forecast_result(x: np.ndarray, n_horizons: int, seed: int = 1):
    """Create a ParametricForecastResult simulating base model forecasts.

    Each horizon adds increasing noise to x.
    """
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    loc = np.column_stack([
        x[:, 0] + rng.normal(0, 0.5 * (h + 1), N)
        for h in range(n_horizons)
    ])
    scale = np.full((N, n_horizons), 1.5)
    return ParametricForecastResult(
        dist_name="normal",
        params={"loc": loc, "scale": scale},
        basis_index=pd.RangeIndex(N),
        model_name="base_model",
    )


# =====================================================================
# E2E Test
# =====================================================================

class TestCKDE2E:
    """Full pipeline: data → search space → Optuna → runner → CRPS."""

    def test_e2e_pipeline(self):
        """Complete adaptive bandwidth pipeline with timing."""
        timings = {}

        # ----- Step 1: Generate data & split -----
        t0 = time.perf_counter()
        N_TOTAL = 500
        N_TRAIN = 300
        N_VAL = 200
        H = 3  # horizons

        x_all, y_all = _make_power_curve_data(N_TOTAL, seed=42)
        x_train, y_train = x_all[:N_TRAIN], y_all[:N_TRAIN]
        x_val, y_val = x_all[N_TRAIN:], y_all[N_TRAIN:]

        val_result = _make_forecast_result(x_val, n_horizons=H, seed=10)
        test_result = _make_forecast_result(x_val, n_horizons=H, seed=20)
        timings["data_generation"] = time.perf_counter() - t0

        # ----- Step 2: Build model to get search space -----
        t0 = time.perf_counter()
        config = CKDConfig(n_x_vars=1, n_samples=200)
        probe_model = ConditionalKernelDensity(config)
        probe_model.build(x_train, y_train, ["wind_speed"])

        x_range = probe_model.x_search_ranges_[0]
        y_range = probe_model.y_search_range_
        timings["search_space_computation"] = time.perf_counter() - t0

        # Verify search space is sensible
        assert x_range[0] > 0, "x search lower bound must be positive"
        assert x_range[1] > x_range[0], "x search range must be valid"
        assert y_range[0] > 0, "y search lower bound must be positive"
        assert y_range[1] > y_range[0], "y search range must be valid"

        # ----- Step 3: Optuna optimization via CKDRunner -----
        t0 = time.perf_counter()
        optuna_config = CKDOptunaConfig(
            n_trials=5,
            decay_range=(0.98, 1.0),
        )
        runner = CKDRunner(
            x_obs=x_train,
            y_obs=y_train,
            x_columns=["wind_speed"],
            base_config=config,
            optuna_config=optuna_config,
        )
        runner.fit_hyperparameters(
            val_results=[val_result],
            val_observed=y_val,
        )
        timings["optuna_optimization"] = time.perf_counter() - t0

        assert runner.is_fitted_
        assert len(runner.models_) == H

        # Check optimized bandwidths are within search range
        for h in range(1, H + 1):
            model = runner.models_[h]
            hp = model.get_hyperparameters()
            x_bw = hp["x_bandwidth"]["wind_speed"]
            y_bw = hp["y_bandwidth"]
            assert x_range[0] <= x_bw <= x_range[1], (
                f"h={h}: x_bw={x_bw} outside range {x_range}"
            )
            assert y_range[0] <= y_bw <= y_range[1], (
                f"h={h}: y_bw={y_bw} outside range {y_range}"
            )

        # ----- Step 4: Run per-horizon prediction -----
        t0 = time.perf_counter()
        result = runner.run(
            test_results=[test_result],
            seed=42,
        )
        timings["prediction"] = time.perf_counter() - t0

        assert isinstance(result, GridForecastResult)
        assert result.prob.shape[0] == N_VAL
        assert result.prob.shape[2] == H

        # ----- Step 5: CRPS evaluation -----
        t0 = time.perf_counter()
        crps_per_horizon = []
        for h in range(1, H + 1):
            gd = result.to_distribution(h)
            crps_vals = grid_crps(gd, y_val)
            assert crps_vals.shape == (N_VAL,)
            assert np.all(np.isfinite(crps_vals))
            assert np.all(crps_vals >= 0)
            crps_per_horizon.append(float(crps_vals.mean()))
        timings["crps_evaluation"] = time.perf_counter() - t0

        # ----- Report -----
        total = sum(timings.values())
        print("\n" + "=" * 60)
        print(f"CKD E2E Pipeline Timing (N_train={N_TRAIN}, N_val={N_VAL}, H={H})")
        print("=" * 60)
        for stage, t in timings.items():
            print(f"  {stage:30s}: {t:8.3f}s")
        print(f"  {'TOTAL':30s}: {total:8.3f}s")
        print(f"\nSearch space (x): [{x_range[0]:.4f}, {x_range[1]:.4f}]")
        print(f"Search space (y): [{y_range[0]:.4f}, {y_range[1]:.4f}]")
        print(f"\nCRPS per horizon: {crps_per_horizon}")
        for h in range(1, H + 1):
            hp = runner.models_[h].get_hyperparameters()
            print(
                f"  h={h}: x_bw={hp['x_bandwidth']['wind_speed']:.4f}, "
                f"y_bw={hp['y_bandwidth']:.4f}, "
                f"decay={hp['time_decay_factor']:.4f}, "
                f"x_pts={hp['x_basis_points']}, "
                f"y_pts={hp['y_basis_points']}"
            )

    def test_e2e_scaling(self):
        """Measure how optimization time scales with data size."""
        sizes = [100, 300]
        timings = {}

        for n in sizes:
            x, y = _make_power_curve_data(n, seed=0)
            n_train = int(n * 0.7)
            x_train, y_train = x[:n_train], y[:n_train]
            x_val, y_val = x[n_train:], y[n_train:]

            val_result = _make_forecast_result(x_val, n_horizons=2, seed=1)

            config = CKDConfig(n_x_vars=1, n_samples=100)
            optuna_config = CKDOptunaConfig(n_trials=3, decay_range=(0.99, 1.0))
            runner = CKDRunner(
                x_obs=x_train, y_obs=y_train, x_columns=["wind_speed"],
                base_config=config, optuna_config=optuna_config,
            )

            t0 = time.perf_counter()
            runner.fit_hyperparameters(
                val_results=[val_result], val_observed=y_val,
            )
            elapsed = time.perf_counter() - t0
            timings[n] = elapsed

        print("\n" + "=" * 60)
        print("CKD Optimization Scaling")
        print("=" * 60)
        for n, t in timings.items():
            print(f"  N={n:5d}: {t:.3f}s")

        # Should complete in reasonable time
        for n, t in timings.items():
            assert t < 60, f"N={n} took {t:.1f}s, too slow"
