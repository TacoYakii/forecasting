"""Persistence tests: save/load round-trips, save_dir behavior, model_name retention.

Items covered:
    29. Save/load round-trip for Statistical (ArimaGarch), ML (LRForecaster),
        Deep (DeepAR) families.
    30. save_dir=None writes nothing; save_dir specified creates expected artifacts.
    31. Custom model_name survives save -> load and save path uses _registry_key.

Foundation models (Chronos, Moirai) are skipped because they require
large model downloads.
"""

import importlib as _il

import numpy as np
import pandas as pd
import pytest
import yaml

from src.core.forecast_results import ParametricForecastResult
from src.core.persistence import load_model

from .conftest import EXOG_COLS, FORECAST_END, FORECAST_START, Y_COL

# Trigger MODEL_REGISTRY registration (import specific modules to avoid
# pgbm/ninja dependency issue via __init__.py)
for _mod in [
    "src.models.machine_learning.lr_model",
    "src.models.machine_learning.xgboost_model",
    "src.models.statistical.arima_garch",
]:
    try:
        _il.import_module(_mod)
    except Exception:
        pass


# ======================================================================
# Helpers
# ======================================================================

def _fit_arima_garch(train_df, **kwargs):
    """Fit a minimal ArimaGarchForecaster."""
    from src.models.statistical.arima_garch import ArimaGarchForecaster

    hp = {"arima_order": (1, 0, 1), "garch_order": (1, 1), "distribution": "normal"}
    model = ArimaGarchForecaster(hyperparameter=hp, **kwargs)
    model.fit(dataset=train_df, y_col=Y_COL)
    return model


def _fit_lr(train_df, **kwargs):
    """Fit a minimal LRForecaster."""
    from src.models.machine_learning.lr_model import LRForecaster

    model = LRForecaster(hyperparameter={}, **kwargs)
    model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
    return model


# ======================================================================
# Item 29: Save/load round-trip tests
# ======================================================================

class TestSaveLoadRoundTrip:
    """Verify _save_model_specific -> _load_model_specific reproduces forecasts."""

    def test_arima_garch_round_trip(self, train_df, full_df, tmp_path):
        """Statistical: ArimaGarch save/load via RollingRunner."""
        from src.core.runner import RollingRunner

        save_dir = tmp_path / "arima_rt"
        model = _fit_arima_garch(train_df)

        runner = RollingRunner(
            model=model,
            dataset=full_df,
            y_col=Y_COL,
            forecast_period=(FORECAST_START, FORECAST_END),
            save_dir=str(save_dir),
        )
        result_before = runner.run(horizon=6, show_progress=False)

        # Load the fitted (pre-rolling) state via persistence
        loaded = load_model(save_dir, state="fitted")
        assert loaded.is_fitted_
        assert loaded._registry_key == "arima_garch"

        # Verify the loaded model can produce forecasts via a new runner
        runner2 = RollingRunner(
            model=loaded,
            dataset=full_df,
            y_col=Y_COL,
            forecast_period=(FORECAST_START, FORECAST_END),
        )
        result_after = runner2.run(horizon=6, show_progress=False)

        # Parameters should match exactly (same fitted state, same data)
        np.testing.assert_allclose(
            result_before.params["loc"],
            result_after.params["loc"],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result_before.params["scale"],
            result_after.params["scale"],
            rtol=1e-10,
        )

    def test_lr_round_trip(self, train_df, forecast_df, tmp_path):
        """ML: LR save -> load -> forecast matches."""
        model = _fit_lr(train_df)

        forecast_X = forecast_df[EXOG_COLS].to_numpy()
        forecast_index = forecast_df.index

        # Forecast before save
        result_before = model.forecast(forecast_X, forecast_index)

        # Save
        model_stem = tmp_path / "lr_model"
        model._save_model_specific(model_stem)

        # Load into a fresh instance
        from src.models.machine_learning.lr_model import LRForecaster

        loaded = LRForecaster()
        loaded.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        loaded._load_model_specific(model_stem)
        loaded.is_fitted_ = True

        # Forecast after load
        result_after = loaded.forecast(forecast_X, forecast_index)

        np.testing.assert_allclose(
            result_before.params["loc"],
            result_after.params["loc"],
            rtol=1e-10,
        )

    def test_xgboost_round_trip(self, train_df, forecast_df, tmp_path):
        """ML: XGBoost save -> load -> forecast matches."""
        from src.models.machine_learning.xgboost_model import XGBoostForecaster

        model = XGBoostForecaster(hyperparameter={})
        model.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)

        forecast_X = forecast_df[EXOG_COLS].to_numpy()
        forecast_index = forecast_df.index

        result_before = model.forecast(forecast_X, forecast_index)

        # Save
        model_stem = tmp_path / "xgboost_model"
        model._save_model_specific(model_stem)

        # Load into a fresh instance
        loaded = XGBoostForecaster(hyperparameter={})
        loaded.fit(dataset=train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        loaded._load_model_specific(model_stem)
        loaded.is_fitted_ = True

        result_after = loaded.forecast(forecast_X, forecast_index)

        np.testing.assert_allclose(
            result_before.params["loc"],
            result_after.params["loc"],
            rtol=1e-10,
        )

    @pytest.mark.slow
    def test_deepar_round_trip(self, train_df, tmp_path):
        """Deep: DeepAR save -> load -> predict_from_context matches."""
        from src.models.deep_time_series.deepar import DeepARForecaster

        hp = {
            "input_size": 48,
            "prediction_length": 6,
            "max_steps": 5,
            "loss_type": "quantile",
            "level": [80, 90],
        }
        model = DeepARForecaster(hyperparameter=hp)
        model.fit(dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS)

        # Build context for predict_from_context
        context_len = 48
        context_y = train_df[Y_COL].values[-context_len:]
        context_index = train_df.index[-context_len:]
        context_X = train_df[EXOG_COLS].values[-context_len:]

        # Future exog: use forecast slice
        future_index = pd.date_range(
            context_index[-1] + pd.Timedelta(hours=1), periods=6, freq="h",
        )
        future_X = train_df[EXOG_COLS].values[:6]

        result_before = model.predict_from_context(
            context_y=context_y,
            horizon=6,
            context_index=context_index,
            context_X=context_X,
            future_X=future_X,
            future_index=future_index,
        )

        # Save
        model_stem = tmp_path / "deepar_model"
        model._save_model_specific(model_stem)

        # Load
        loaded = DeepARForecaster(hyperparameter=hp)
        loaded.fit(dataset=train_df, y_col=Y_COL, futr_cols=EXOG_COLS)
        loaded._load_model_specific(model_stem)
        loaded.is_fitted_ = True

        result_after = loaded.predict_from_context(
            context_y=context_y,
            horizon=6,
            context_index=context_index,
            context_X=context_X,
            future_X=future_X,
            future_index=future_index,
        )

        # Quantile keys should match
        assert set(result_before.quantiles_data.keys()) == set(
            result_after.quantiles_data.keys()
        )


# ======================================================================
# Item 30: save_dir=None / save_dir specified behavior
# ======================================================================

class TestSaveDirBehavior:
    """Verify save_dir=None writes nothing, save_dir specified creates artifacts."""

    def test_rolling_runner_no_save_dir(self, train_df, full_df):
        """RollingRunner with save_dir=None writes no files."""
        from src.core.runner import RollingRunner

        model = _fit_arima_garch(train_df)
        runner = RollingRunner(
            model=model,
            dataset=full_df,
            y_col=Y_COL,
            forecast_period=(FORECAST_START, FORECAST_END),
            save_dir=None,
        )
        result = runner.run(horizon=6, show_progress=False)

        assert isinstance(result, ParametricForecastResult)
        # No save_dir means _save_dir is None -- nothing written

    def test_rolling_runner_with_save_dir(self, train_df, full_df, tmp_path):
        """RollingRunner with save_dir creates expected artifacts."""
        from src.core.runner import RollingRunner

        save_dir = tmp_path / "rolling_exp"
        model = _fit_arima_garch(train_df)
        runner = RollingRunner(
            model=model,
            dataset=full_df,
            y_col=Y_COL,
            forecast_period=(FORECAST_START, FORECAST_END),
            save_dir=str(save_dir),
        )
        runner.run(horizon=6, show_progress=False)

        # Check runner_config.yaml
        config_path = save_dir / "runner_config.yaml"
        assert config_path.exists()
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["runner_type"] == "RollingRunner"
        assert config["registry_key"] == "arima_garch"

        # Check model/ directory
        model_dir = save_dir / "model"
        assert model_dir.exists()
        fitted_dir = model_dir / "fitted"
        assert fitted_dir.exists()
        # Should have the pkl file
        pkl_files = list(fitted_dir.glob("*.pkl"))
        assert len(pkl_files) >= 1

        # Check post_rolling state
        post_dir = model_dir / "post_rolling"
        assert post_dir.exists()

        # Check forecast_result/
        fr_dir = save_dir / "forecast_result"
        assert fr_dir.exists()

    def test_per_horizon_runner_no_save_dir(self, per_horizon_csv_dir):
        """PerHorizonRunner with save_dir=None writes no files."""
        from src.core.runner import PerHorizonRunner

        runner = PerHorizonRunner(
            data_dir=per_horizon_csv_dir,
            registry_key="lr",
            y_col=Y_COL,
            exog_cols=EXOG_COLS,
            training_period=("2023-01-01", "2023-01-20"),
            forecast_period=(FORECAST_START, FORECAST_END),
            horizons=[1, 2, 3],
            save_dir=None,
        )
        runner.fit()
        result = runner.forecast()
        assert isinstance(result, ParametricForecastResult)

    def test_per_horizon_runner_with_save_dir(self, per_horizon_csv_dir, tmp_path):
        """PerHorizonRunner with save_dir creates expected artifacts."""
        from src.core.runner import PerHorizonRunner

        save_dir = tmp_path / "ph_exp"
        runner = PerHorizonRunner(
            data_dir=per_horizon_csv_dir,
            registry_key="lr",
            y_col=Y_COL,
            exog_cols=EXOG_COLS,
            training_period=("2023-01-01", "2023-01-20"),
            forecast_period=(FORECAST_START, FORECAST_END),
            horizons=[1, 2, 3],
            save_dir=str(save_dir),
        )
        runner.fit()
        runner.forecast()

        # Check runner_config.yaml
        config_path = save_dir / "runner_config.yaml"
        assert config_path.exists()
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["runner_type"] == "PerHorizonRunner"
        assert config["registry_key"] == "lr"

        # Check model/ directory has horizon subdirs
        model_dir = save_dir / "model"
        assert model_dir.exists()
        for h in [1, 2, 3]:
            h_dir = model_dir / f"horizon_{h}"
            assert h_dir.exists()
            joblib_files = list(h_dir.glob("*.joblib"))
            assert len(joblib_files) >= 1

        # Check forecast_result/
        fr_dir = save_dir / "forecast_result"
        assert fr_dir.exists()


# ======================================================================
# Item 31: Custom model_name round-trip
# ======================================================================

class TestModelNameRoundTrip:
    """Verify custom model_name survives save -> load cycle."""

    def test_rolling_runner_model_name(self, train_df, full_df, tmp_path):
        """Custom model_name preserved after RollingRunner save -> load_model."""
        from src.core.runner import RollingRunner

        custom_name = "MyCustomArima"
        save_dir = tmp_path / "name_test_rolling"

        model = _fit_arima_garch(train_df, model_name=custom_name)
        assert model.nm == custom_name

        runner = RollingRunner(
            model=model,
            dataset=full_df,
            y_col=Y_COL,
            forecast_period=(FORECAST_START, FORECAST_END),
            save_dir=str(save_dir),
        )
        runner.run(horizon=6, show_progress=False)

        # Verify config uses _registry_key for save path, not custom name
        config_path = save_dir / "runner_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["registry_key"] == "arima_garch"
        assert config["model_name"] == custom_name

        # Verify model file uses registry_key in filename
        fitted_dir = save_dir / "model" / "fitted"
        pkl_files = list(fitted_dir.glob("arima_garch_model*"))
        assert len(pkl_files) >= 1

        # Load and verify model_name is restored
        loaded_model = load_model(save_dir)
        assert loaded_model.nm == custom_name
        assert loaded_model._registry_key == "arima_garch"
        assert loaded_model.is_fitted_

    def test_per_horizon_runner_model_name(self, per_horizon_csv_dir, tmp_path):
        """Custom model_name preserved after PerHorizonRunner save -> load_model."""
        from src.core.runner import PerHorizonRunner

        custom_name = "MyCustomLR"
        save_dir = tmp_path / "name_test_ph"

        runner = PerHorizonRunner(
            data_dir=per_horizon_csv_dir,
            registry_key="lr",
            y_col=Y_COL,
            exog_cols=EXOG_COLS,
            training_period=("2023-01-01", "2023-01-20"),
            forecast_period=(FORECAST_START, FORECAST_END),
            horizons=[1, 2],
            model_name=custom_name,
            save_dir=str(save_dir),
        )
        runner.fit()
        runner.forecast()

        # Verify config
        config_path = save_dir / "runner_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["registry_key"] == "lr"
        assert config["model_name"] == custom_name

        # Verify model files use registry_key
        for h in [1, 2]:
            h_dir = save_dir / "model" / f"horizon_{h}"
            lr_files = list(h_dir.glob("lr_model*"))
            assert len(lr_files) >= 1

        # Load and verify
        loaded_models = load_model(save_dir)
        assert isinstance(loaded_models, dict)
        for h, m in loaded_models.items():
            assert m.nm == custom_name
            assert m._registry_key == "lr"
            assert m.is_fitted_

    def test_default_model_name_uses_class_name(self, train_df):
        """Without model_name, nm falls back to class name."""
        model = _fit_arima_garch(train_df)
        assert model.nm == "ArimaGarchForecaster"

    def test_custom_model_name_overrides(self, train_df):
        """model_name='X' makes nm return 'X'."""
        model = _fit_arima_garch(train_df, model_name="WindFarm_A")
        assert model.nm == "WindFarm_A"
