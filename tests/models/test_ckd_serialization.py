"""Tests for CKD model and runner serialization (save/load).

Verifies that fitted ConditionalKernelDensity and CKDRunner can be
persisted to disk and restored with identical apply() output.
"""

import numpy as np
import pytest
from pathlib import Path

from src.models.conditional_kernel_density import (
    CKDConfig,
    ConditionalKernelDensity,
)
from src.core.ckd_runner import CKDRunner


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def fitted_model(synthetic_df):
    """Fitted CKD model with 1 explanatory variable."""
    df = synthetic_df.iloc[:300]
    x = df[["wind_speed"]].values
    y = df["power"].values
    config = CKDConfig(n_x_vars=1, n_samples=200)
    model = ConditionalKernelDensity(config)
    model.build(x, y, ["wind_speed"])
    return model


@pytest.fixture
def test_samples(synthetic_df):
    """Test input samples for apply()."""
    rng = np.random.default_rng(42)
    ws = synthetic_df["wind_speed"].iloc[300:350].values
    return [ws[:, None] + rng.normal(0, 0.5, (50, 200))]


# =====================================================================
# ConditionalKernelDensity save/load
# =====================================================================

class TestCKDSerialization:
    """Tests for ConditionalKernelDensity.save() and .load()."""

    def test_save_creates_pkl(self, fitted_model, tmp_path):
        """save() creates a .pkl file."""
        path = fitted_model.save(tmp_path / "model")
        assert path.suffix == ".pkl"
        assert path.exists()

    def test_save_adds_suffix(self, fitted_model, tmp_path):
        """save() adds .pkl suffix when not provided."""
        path = fitted_model.save(tmp_path / "model.dat")
        assert path.suffix == ".pkl"

    def test_save_unfitted_raises(self, tmp_path):
        """save() raises RuntimeError on unfitted model."""
        model = ConditionalKernelDensity(CKDConfig(n_x_vars=1))
        with pytest.raises(RuntimeError, match="unfitted"):
            model.save(tmp_path / "bad")

    def test_load_restores_fitted_state(self, fitted_model, tmp_path):
        """load() restores is_fitted and all attributes."""
        fitted_model.save(tmp_path / "model")
        loaded = ConditionalKernelDensity.load(tmp_path / "model")

        assert loaded.is_fitted is True
        assert loaded.x_columns == fitted_model.x_columns
        assert loaded.n_x == fitted_model.n_x
        assert loaded.x_bandwidths == fitted_model.x_bandwidths
        assert loaded.y_bandwidth == fitted_model.y_bandwidth
        assert loaded.time_decay_factor == fitted_model.time_decay_factor

    def test_load_density_exact(self, fitted_model, tmp_path):
        """Density tensor is bit-for-bit identical after load."""
        fitted_model.save(tmp_path / "model")
        loaded = ConditionalKernelDensity.load(tmp_path / "model")

        np.testing.assert_array_equal(loaded.density, fitted_model.density)
        np.testing.assert_array_equal(loaded.y_basis, fitted_model.y_basis)
        for orig, rest in zip(fitted_model.x_basis_list, loaded.x_basis_list):
            np.testing.assert_array_equal(rest, orig)

    def test_apply_identical_after_load(
        self, fitted_model, test_samples, tmp_path,
    ):
        """apply() produces identical output before and after save/load."""
        gd_orig = fitted_model.apply(test_samples, seed=0)

        fitted_model.save(tmp_path / "model")
        loaded = ConditionalKernelDensity.load(tmp_path / "model")
        gd_loaded = loaded.apply(test_samples, seed=0)

        np.testing.assert_array_equal(gd_loaded.prob, gd_orig.prob)
        np.testing.assert_array_equal(gd_loaded.grid, gd_orig.grid)

    def test_config_preserved(self, fitted_model, tmp_path):
        """Config fields survive round-trip."""
        fitted_model.save(tmp_path / "model")
        loaded = ConditionalKernelDensity.load(tmp_path / "model")

        assert loaded.config.n_x_vars == fitted_model.config.n_x_vars
        assert loaded.config.bins_per_bandwidth == fitted_model.config.bins_per_bandwidth
        assert loaded.config.n_samples == fitted_model.config.n_samples

    def test_stats_preserved(self, fitted_model, tmp_path):
        """Data statistics and search ranges survive round-trip."""
        fitted_model.save(tmp_path / "model")
        loaded = ConditionalKernelDensity.load(tmp_path / "model")

        assert loaded.x_stats_ == fitted_model.x_stats_
        assert loaded.y_stats_ == fitted_model.y_stats_
        assert loaded.x_search_ranges_ == fitted_model.x_search_ranges_
        assert loaded.y_search_range_ == fitted_model.y_search_range_

    def test_injected_bandwidths_flag(self, synthetic_df, tmp_path):
        """_bandwidths_injected flag is preserved after round-trip."""
        df = synthetic_df.iloc[:300]
        x = df[["wind_speed"]].values
        y = df["power"].values
        config = CKDConfig(n_x_vars=1)
        model = ConditionalKernelDensity(config)
        model.set_bandwidths([1.5], 0.3)
        model.build(x, y, ["wind_speed"])

        model.save(tmp_path / "injected")
        loaded = ConditionalKernelDensity.load(tmp_path / "injected")

        assert loaded._bandwidths_injected is True
        assert loaded.x_bandwidths == [1.5]
        assert loaded.y_bandwidth == 0.3


# =====================================================================
# CKDRunner save/load
# =====================================================================

class TestCKDRunnerSerialization:
    """Tests for CKDRunner.save() and .load()."""

    def test_save_unfitted_raises(self, synthetic_df, tmp_path):
        """save() raises RuntimeError on unfitted runner."""
        df = synthetic_df.iloc[:300]
        runner = CKDRunner(
            x_obs=df[["wind_speed"]].values,
            y_obs=df["power"].values,
            x_columns=["wind_speed"],
            base_config=CKDConfig(n_x_vars=1),
        )
        with pytest.raises(RuntimeError, match="unfitted"):
            runner.save(tmp_path / "bad")

    def test_shared_model_round_trip(self, synthetic_df, tmp_path):
        """Runner with shared model (no Optuna) survives round-trip."""
        df = synthetic_df.iloc[:300]
        runner = CKDRunner(
            x_obs=df[["wind_speed"]].values,
            y_obs=df["power"].values,
            x_columns=["wind_speed"],
            base_config=CKDConfig(n_x_vars=1, n_samples=200),
        )
        runner.fit_hyperparameters()

        path = runner.save(tmp_path / "runner")
        loaded = CKDRunner.load(path)

        assert loaded.is_fitted_ is True
        assert loaded.x_columns == ["wind_speed"]
        assert loaded.model_name == runner.model_name
        assert hasattr(loaded, "_shared_model")
        assert loaded._shared_model.is_fitted

    def test_per_horizon_round_trip(self, synthetic_df, tmp_path):
        """Runner with per-horizon models survives round-trip."""
        df = synthetic_df.iloc[:300]
        x = df[["wind_speed"]].values
        y = df["power"].values

        # Manually create per-horizon models to avoid Optuna dependency
        config = CKDConfig(n_x_vars=1, n_samples=200)
        runner = CKDRunner(
            x_obs=x, y_obs=y,
            x_columns=["wind_speed"],
            base_config=config,
        )
        # Simulate per-horizon fitting
        for h in range(1, 4):
            model = ConditionalKernelDensity(config)
            model.build(x, y, ["wind_speed"])
            runner.models_[h] = model
        runner.is_fitted_ = True

        path = runner.save(tmp_path / "runner_ph")
        loaded = CKDRunner.load(path)

        assert loaded.is_fitted_ is True
        assert set(loaded.models_.keys()) == {1, 2, 3}
        for h in range(1, 4):
            assert loaded.models_[h].is_fitted
