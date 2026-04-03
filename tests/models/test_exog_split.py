"""Tests for exogenous variable classification (futr_exog / hist_exog).

Verifies that:
1. Synthetic data with futr_exog, hist_exog, and target columns is correctly
   handled by each model family.
2. Deep models properly split futr/hist to NeuralForecast.
3. Statistical models use only futr_exog (no hist leakage).

Forecast tests save plots to tests/models/plots/ for visual inspection.
Run with: uv run pytest tests/models/test_exog_classification.py -v -m "not slow"
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.forecast_results import ParametricForecastResult


def _result_mean(result):
    """Reconstruct (N, H) mean array from per-horizon distributions."""
    return np.column_stack([
        result.to_distribution(h).mean() for h in range(1, result.horizon + 1)
    ])


def _result_std(result):
    """Reconstruct (N, H) std array from per-horizon distributions."""
    return np.column_stack([
        result.to_distribution(h).std() for h in range(1, result.horizon + 1)
    ])


# ======================================================================
# Synthetic data: futr_exog + hist_exog + target
# ======================================================================

FUTR_COLS = ["nwp_wspd", "nwp_temp"]
HIST_COLS = ["obs_wspd"]
EXOG_COLS = FUTR_COLS + HIST_COLS
Y_COL = "power"

TRAIN_END = "2023-01-20"
FORECAST_START = "2023-01-21"
FORECAST_END = "2023-01-25"
HORIZON = 6

PLOT_DIR = Path(__file__).parent / "plots"


def _generate_futr_hist_data(
    n_hours: int = 720,
    start: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data with futr_exog, hist_exog, and target.

    Columns:
        nwp_wspd:  NWP wind speed forecast (futr_exog) - known in advance
        nwp_temp:  NWP temperature forecast (futr_exog) - known in advance
        obs_wspd:  Observed wind speed (hist_exog) - only past available
        power:     Wind power output (target)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    # Actual wind speed (ground truth)
    actual_wspd = 7.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1.0, n_hours)
    actual_wspd = np.maximum(actual_wspd, 0.5)

    # NWP forecast wind speed (futr_exog) = actual + forecast error
    nwp_wspd = actual_wspd + rng.normal(0, 1.5, n_hours)
    nwp_wspd = np.maximum(nwp_wspd, 0.0)

    # NWP temperature (futr_exog)
    nwp_temp = 15.0 + 5.0 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 1.0, n_hours)

    # Observed wind speed (hist_exog) = actual (only available for past)
    obs_wspd = actual_wspd

    # Power: depends on actual wind speed + noise
    power = 5.0 + 1.5 * actual_wspd**1.3 - 0.2 * nwp_temp + rng.normal(0, 2.0, n_hours)
    power = np.clip(power, 0.5, 100.0)

    index = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame(
        {
            "nwp_wspd": nwp_wspd,
            "nwp_temp": nwp_temp,
            "obs_wspd": obs_wspd,
            "power": power,
        },
        index=index,
    )


# ======================================================================
# Plot helper
# ======================================================================

def _save_forecast_plot(
    train_df: pd.DataFrame,
    forecast_index: pd.Index,
    observed: np.ndarray,
    predicted_mu: np.ndarray,
    predicted_std: np.ndarray,
    title: str,
    filename: str,
):
    """Save a plot showing training data, observed forecast period, and predictions.

    Top panel: target (y) with train/forecast split + prediction intervals.
    Bottom panel: exogenous variables over the same period.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Show last 48h of training + forecast period
    plot_start = pd.Timestamp(FORECAST_START) - pd.Timedelta(hours=48)
    train_plot = train_df.loc[plot_start:TRAIN_END]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Top: target + predictions ---
    ax = axes[0]
    ax.plot(train_plot.index, train_plot[Y_COL], "k-", label="Train (observed)", alpha=0.7)
    ax.plot(forecast_index, observed, "b-", label="Forecast period (observed)", linewidth=1.5)
    ax.plot(forecast_index, predicted_mu, "r--", label="Predicted (mu)", linewidth=1.5)
    if predicted_std is not None:
        ax.fill_between(
            forecast_index,
            predicted_mu - 1.96 * predicted_std,
            predicted_mu + 1.96 * predicted_std,
            color="red", alpha=0.15, label="95% CI",
        )
    ax.axvline(pd.Timestamp(FORECAST_START), color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel(Y_COL)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    # --- Bottom: exogenous variables ---
    ax2 = axes[1]
    all_plot = pd.concat([train_plot, train_df.loc[FORECAST_START:].head(len(forecast_index))])
    for col in FUTR_COLS:
        if col in all_plot.columns:
            ax2.plot(all_plot.index, all_plot[col], label=f"{col} (futr)", alpha=0.7)
    for col in HIST_COLS:
        if col in all_plot.columns:
            ax2.plot(all_plot.index, all_plot[col], "--", label=f"{col} (hist)", alpha=0.7)
    ax2.axvline(pd.Timestamp(FORECAST_START), color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Exogenous")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=100)
    plt.close(fig)


@pytest.fixture(scope="module")
def exog_df() -> pd.DataFrame:
    return _generate_futr_hist_data()


@pytest.fixture()
def exog_train_df(exog_df) -> pd.DataFrame:
    return exog_df.loc[:TRAIN_END].copy()


@pytest.fixture()
def exog_forecast_df(exog_df) -> pd.DataFrame:
    return exog_df.loc[FORECAST_START:FORECAST_END].copy()


@pytest.fixture()
def exog_full_df(exog_df) -> pd.DataFrame:
    """Full dataset for Deep models that need context + future."""
    return exog_df.loc[:FORECAST_END].copy()


# ======================================================================
# Statistical models: exog_cols only (futr only)
# ======================================================================

class TestStatisticalExog:
    """Statistical models receive only futr_exog via exog_cols."""

    def test_arima_garch_futr_only(self, exog_df, exog_train_df, exog_forecast_df, tmp_path):
        """ARIMA-GARCH trained with futr_exog only, forecast with x_future."""
        from src.models.statistical.arima_garch import ArimaGarchForecaster

        model = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1), "garch_order": (1, 1), "distribution": "normal",
        })
        model.fit(dataset=exog_train_df, y_col=Y_COL, exog_cols=FUTR_COLS)
        assert model.is_fitted_

        x_future = exog_forecast_df[FUTR_COLS].to_numpy()[:HORIZON]
        result = model.forecast(horizon=HORIZON, x_future=x_future)
        assert isinstance(result, ParametricForecastResult)
        assert result.params["loc"].shape == (1, HORIZON)
        assert np.all(np.isfinite(_result_mean(result)))

        # Plot
        forecast_index = exog_forecast_df.index[:HORIZON]
        observed = exog_forecast_df[Y_COL].to_numpy()[:HORIZON]
        _save_forecast_plot(
            train_df=exog_df.loc[:TRAIN_END],
            forecast_index=forecast_index,
            observed=observed,
            predicted_mu=_result_mean(result).ravel(),
            predicted_std=_result_std(result).ravel(),
            title="ARIMA-GARCH | exog_cols=FUTR only",
            filename="statistical_arima_garch_futr.png",
        )

    def test_arima_garch_update_state_with_exog(self, exog_train_df, exog_forecast_df, tmp_path):
        """update_state with current exog, then forecast with future exog."""
        from src.models.statistical.arima_garch import ArimaGarchForecaster

        model = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1), "garch_order": (1, 1), "distribution": "normal",
        })
        model.fit(dataset=exog_train_df, y_col=Y_COL, exog_cols=FUTR_COLS)

        y_new = exog_forecast_df[Y_COL].iloc[0]
        x_new = exog_forecast_df[FUTR_COLS].iloc[0].to_numpy()
        model.update_state(y_new, x_new)

        x_future = exog_forecast_df[FUTR_COLS].to_numpy()[1:HORIZON + 1]
        result = model.forecast(horizon=HORIZON, x_future=x_future)
        assert isinstance(result, ParametricForecastResult)
        assert np.all(np.isfinite(_result_mean(result)))

    def test_arima_garch_exog_shape(self, exog_train_df, tmp_path):
        """Verify model stores correct number of exog features."""
        from src.models.statistical.arima_garch import ArimaGarchForecaster

        model = ArimaGarchForecaster(hyperparameter={
            "arima_order": (1, 0, 1), "garch_order": (1, 1), "distribution": "normal",
        })
        model.fit(dataset=exog_train_df, y_col=Y_COL, exog_cols=FUTR_COLS)

        assert model.exog_cols == FUTR_COLS
        assert model.X.shape[1] == len(FUTR_COLS)


# ======================================================================
# ML models: exog_cols (all features from CSV)
# ======================================================================

class TestMLExog:
    """ML models receive all exog via exog_cols."""

    @pytest.fixture(params=["lr", "xgboost"])
    def model_name(self, request):
        return request.param

    def test_fit_forecast_with_exog(self, model_name, exog_df, exog_train_df, exog_forecast_df, tmp_path):
        """ML model trained with all exog, forecast with same columns."""
        if model_name == "lr":
            from src.models.machine_learning.lr_model import LRForecaster as model_cls
        elif model_name == "xgboost":
            from src.models.machine_learning.xgboost_model import XGBoostForecaster as model_cls
        else:
            from src.core.registry import MODEL_REGISTRY
            model_cls = MODEL_REGISTRY.get(model_name)
        model = model_cls()
        model.fit(dataset=exog_train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        assert model.is_fitted_

        forecast_X = exog_forecast_df[EXOG_COLS].to_numpy()
        result = model.forecast(forecast_X, exog_forecast_df.index)

        assert isinstance(result, ParametricForecastResult)
        T = len(exog_forecast_df)
        assert result.params["loc"].shape == (T, 1)
        mu = _result_mean(result)
        sigma = _result_std(result)
        assert np.all(np.isfinite(mu))
        assert np.all(mu > -50)
        assert np.all(mu < 200)

        # Plot
        _save_forecast_plot(
            train_df=exog_df.loc[:TRAIN_END],
            forecast_index=exog_forecast_df.index,
            observed=exog_forecast_df[Y_COL].to_numpy(),
            predicted_mu=mu.ravel(),
            predicted_std=sigma.ravel(),
            title=f"{model_name.upper()} | exog_cols=FUTR+HIST",
            filename=f"ml_{model_name}_exog.png",
        )

    def test_futr_only_vs_futr_hist(self, exog_df, exog_train_df, exog_forecast_df, tmp_path):
        """ML model with futr+hist features differs from futr-only."""
        from src.models.machine_learning.lr_model import LRForecaster

        # futr only
        model_futr = LRForecaster()
        model_futr.fit(dataset=exog_train_df, y_col=Y_COL, exog_cols=FUTR_COLS)
        result_futr = model_futr.forecast(
            exog_forecast_df[FUTR_COLS].to_numpy(), exog_forecast_df.index,
        )

        # futr + hist
        model_all = LRForecaster()
        model_all.fit(dataset=exog_train_df, y_col=Y_COL, exog_cols=EXOG_COLS)
        result_all = model_all.forecast(
            exog_forecast_df[EXOG_COLS].to_numpy(), exog_forecast_df.index,
        )

        # Both should be valid
        mu_futr = _result_mean(result_futr)
        mu_all = _result_mean(result_all)
        assert np.all(np.isfinite(mu_futr))
        assert np.all(np.isfinite(mu_all))
        # But predictions should differ (hist adds information)
        assert not np.allclose(mu_futr, mu_all)

        # Plot: compare futr-only vs futr+hist
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 5))

        plot_start = pd.Timestamp(FORECAST_START) - pd.Timedelta(hours=48)
        train_plot = exog_df.loc[plot_start:TRAIN_END]
        ax.plot(train_plot.index, train_plot[Y_COL], "k-", label="Train", alpha=0.5)
        ax.plot(exog_forecast_df.index, exog_forecast_df[Y_COL], "b-", label="Observed", linewidth=1.5)
        ax.plot(exog_forecast_df.index, mu_futr.ravel(), "r--", label="LR (futr only)")
        ax.plot(exog_forecast_df.index, mu_all.ravel(), "g--", label="LR (futr+hist)")
        ax.axvline(pd.Timestamp(FORECAST_START), color="gray", linestyle=":", alpha=0.5)
        ax.set_title("LR | futr-only vs futr+hist comparison")
        ax.set_ylabel(Y_COL)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "ml_lr_futr_vs_futr_hist.png", dpi=100)
        plt.close(fig)

    def test_ml_exog_cols_stored(self, exog_train_df, tmp_path):
        """ML model stores exog_cols correctly."""
        from src.models.machine_learning.lr_model import LRForecaster

        model = LRForecaster()
        model.fit(dataset=exog_train_df, y_col=Y_COL, exog_cols=EXOG_COLS)

        assert model.exog_cols == EXOG_COLS
        assert model.X.shape[1] == len(EXOG_COLS)


# ======================================================================
# Deep models: futr_cols + hist_cols split
# ======================================================================

@pytest.mark.slow
class TestDeepFutrOnly:
    """DeepAR: futr_cols only (no hist_exog support)."""

    _FAST_HP = {
        "prediction_length": HORIZON,
        "max_steps": 5,
        "batch_size": 16,
        "input_size": 24,
    }

    def test_deepar_futr_only(self, exog_df, exog_train_df, exog_full_df, tmp_path):
        """DeepAR with futr_cols only works correctly."""
        from src.models.deep_time_series.deepar import DeepARForecaster

        hp = {**self._FAST_HP, "loss_type": "distribution", "distribution": "Normal"}
        model = DeepARForecaster(hyperparameter=hp)
        model.fit(dataset=exog_train_df, y_col=Y_COL, futr_cols=FUTR_COLS)

        future_df = exog_full_df.loc[FORECAST_START:]
        future_X = future_df[FUTR_COLS].to_numpy()[:HORIZON]
        future_index = future_df.index[:HORIZON]

        result = model.forecast(future_X=future_X, future_index=future_index)
        assert isinstance(result, ParametricForecastResult)
        mu = _result_mean(result)
        sigma = _result_std(result)
        assert np.all(np.isfinite(mu))
        assert np.all(mu > -50)
        assert np.all(mu < 200)

    def test_deepar_hist_cols_ignored_with_warning(self, exog_train_df):
        """DeepAR ignores hist_cols with a warning."""
        import warnings
        from src.models.deep_time_series.deepar import DeepARForecaster

        hp = {**self._FAST_HP, "loss_type": "distribution", "distribution": "Normal"}
        model = DeepARForecaster(hyperparameter=hp)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(dataset=exog_train_df, y_col=Y_COL,
                      futr_cols=FUTR_COLS, hist_cols=HIST_COLS)

            hist_warnings = [x for x in w if "hist_cols" in str(x.message)]
            assert len(hist_warnings) == 1
            assert "does not support" in str(hist_warnings[0].message)

        # hist_cols should be empty after fit (ignored)
        assert model.hist_cols == []
        assert model.futr_cols == FUTR_COLS


@pytest.mark.slow
class TestDeepFutrHist:
    """TFT: futr_cols + hist_cols (supports EXOGENOUS_HIST)."""

    _FAST_HP = {
        "prediction_length": HORIZON,
        "max_steps": 5,
        "batch_size": 16,
        "input_size": 24,
    }

    def test_tft_futr_hist_split(self, exog_train_df):
        """TFT receives futr_cols and hist_cols separately."""
        from src.models.deep_time_series.tft import TFTForecaster

        hp = {**self._FAST_HP, "loss_type": "distribution", "distribution": "Normal"}
        model = TFTForecaster(hyperparameter=hp)
        model.fit(dataset=exog_train_df, y_col=Y_COL,
                  futr_cols=FUTR_COLS, hist_cols=HIST_COLS)

        assert model.futr_cols == FUTR_COLS
        assert model.hist_cols == HIST_COLS
        assert model.exog_cols == FUTR_COLS + HIST_COLS

    def test_tft_futr_hist_fit_forecast(self, exog_df, exog_train_df, exog_full_df):
        """TFT with futr+hist: fit and forecast produce valid result."""
        from src.models.deep_time_series.tft import TFTForecaster

        hp = {**self._FAST_HP, "loss_type": "distribution", "distribution": "Normal"}
        model = TFTForecaster(hyperparameter=hp)
        model.fit(dataset=exog_train_df, y_col=Y_COL,
                  futr_cols=FUTR_COLS, hist_cols=HIST_COLS)
        assert model.is_fitted_

        future_df = exog_full_df.loc[FORECAST_START:]
        future_X = future_df[FUTR_COLS].to_numpy()[:HORIZON]
        future_index = future_df.index[:HORIZON]

        result = model.forecast(future_X=future_X, future_index=future_index)
        assert isinstance(result, ParametricForecastResult)
        assert result.params["loc"].shape == (1, HORIZON)
        mu = _result_mean(result)
        assert np.all(np.isfinite(mu))

    def test_tft_predict_from_context(self, exog_full_df, exog_train_df):
        """TFT predict_from_context with futr+hist context, futr-only future."""
        from src.models.deep_time_series.tft import TFTForecaster

        hp = {**self._FAST_HP, "loss_type": "distribution", "distribution": "Normal"}
        model = TFTForecaster(hyperparameter=hp)
        model.fit(dataset=exog_train_df, y_col=Y_COL,
                  futr_cols=FUTR_COLS, hist_cols=HIST_COLS)

        context_data = exog_full_df.loc[:TRAIN_END]
        context_y = context_data[Y_COL].to_numpy()
        context_index = context_data.index
        all_context_cols = FUTR_COLS + HIST_COLS
        context_X = context_data[all_context_cols].to_numpy()

        future_data = exog_full_df.loc[FORECAST_START:]
        future_X = future_data[FUTR_COLS].to_numpy()[:HORIZON]
        future_index = future_data.index[:HORIZON]

        result = model.predict_from_context(
            context_y=context_y,
            context_index=context_index,
            horizon=HORIZON,
            context_X=context_X,
            future_X=future_X,
            future_index=future_index,
        )
        assert isinstance(result, ParametricForecastResult)
        assert result.params["loc"].shape == (1, HORIZON)
        mu = _result_mean(result)
        assert np.all(np.isfinite(mu))
