"""Run ML / Deep model forecasts for all farms and CV periods.

Usage:
    uv run python scripts/run_ml_forecast.py
    uv run python scripts/run_ml_forecast.py --farms sinan dongbok
    uv run python scripts/run_ml_forecast.py --models ngboost_normal catboost deepar_normal tft_normal
    uv run python scripts/run_ml_forecast.py --n-jobs 8
"""

import argparse
import json
import re
import warnings

warnings.filterwarnings("ignore")
import time
from pathlib import Path

import pandas as pd

import src.models.deep_time_series  # noqa: F401  (register deep models)
import src.models.machine_learning  # noqa: F401  (register ML models)
from src.core.forecast_results import _save_forecast_result
from src.core.runner import PerHorizonRunner, RollingRunner

# ── paths ─────────────────────────────────────────────────────────────
DATA_ROOT = Path("data/training_data_new")
RESULT_ROOT = Path("/home/taco/Documents/windpower_forecasting/new_forecast_res")
PERIOD_JSON = Path("data/meta/modeling/CV_period_setting.json")

# ── per-horizon ML constants ──────────────────────────────────────────
ML_Y_COL = "forecast_time_observed_KPX_pwr"
ML_EXCLUDE_PATTERNS = [
    re.compile(r"^forecast_time_observed_"),
    re.compile(r"^is_valid$"),
]

# ── continuous deep model constants ───────────────────────────────────
DEEP_Y_COL = "observed_KPX_pwr"
DEEP_HORIZON = 48

# ── model configs ─────────────────────────────────────────────────────
# "runner": "per_horizon" uses PerHorizonRunner, "rolling" uses RollingRunner.
MODEL_CONFIGS: dict[str, dict] = {
    # --- ML models (per-horizon) ---
    "ngboost_normal": {
        "runner": "per_horizon",
        "registry_key": "ngboost",
        "hyperparameter": {"Dist": "normal"},
        "dist_name": "normal",
    },
"catboost": {
        "runner": "per_horizon",
        "registry_key": "catboost",
        "hyperparameter": {
            "loss_function": "RMSEWithUncertainty",
            "posterior_sampling": True,
            "silent": True,
            "allow_writing_files": False,
        },
        "dist_name": "normal",
    },
    "pgbm": {
        "runner": "per_horizon",
        "registry_key": "pgbm",
        "hyperparameter": {"Dist": "normal"},
        "dist_name": "normal",
    },
    # --- Deep models (rolling) ---
    "deepar_normal": {
        "runner": "rolling",
        "registry_key": "deepar",
        "hyperparameter": {
            "prediction_length": DEEP_HORIZON,
            "input_size": 3 * DEEP_HORIZON,
            "loss_type": "distribution",
            "distribution": "Normal",
            "max_steps": 100000,
            "batch_size": 32,
            "windows_batch_size": 128,
            "val_size": 336,
            "early_stop_patience_steps": 1000,
            "scaler_type": "robust",
            "level": list(range(2, 100, 2)),
        },
    },
    "tft_normal": {
        "runner": "rolling",
        "registry_key": "tft",
        "hyperparameter": {
            "prediction_length": DEEP_HORIZON,
            "input_size": 3 * DEEP_HORIZON,
            "loss_type": "distribution",
            "distribution": "Normal",
            "max_steps": 100000,
            "batch_size": 64,
            "val_size": 336,
            "early_stop_patience_steps": 1000,
            "scaler_type": "robust",
            "level": list(range(2, 100, 2)),
        },
    },
}


# ── helpers ───────────────────────────────────────────────────────────


def _resolve_ml_exog_cols(data_dir: Path) -> list[str]:
    """Read column names from horizon_1.csv and exclude target/observed/is_valid."""
    sample = pd.read_csv(data_dir / "horizon_1.csv", nrows=0)
    return [
        c
        for c in sample.columns
        if c not in ("basis_time", "forecast_time", ML_Y_COL)
        and not any(p.match(c) for p in ML_EXCLUDE_PATTERNS)
    ]


def _resolve_deep_cols(data_dir: Path) -> tuple[list[str], list[str]]:
    """Resolve futr_cols and hist_cols from continuous farm_level.csv."""
    sample = pd.read_csv(data_dir / "farm_level.csv", nrows=0)
    futr_cols = [c for c in sample.columns if c.startswith(("ECMWF_", "KMA_"))]
    hist_cols = [
        c
        for c in sample.columns
        if c.startswith("observed_") and c != DEEP_Y_COL
    ]
    return futr_cols, hist_cols


# ── run functions ─────────────────────────────────────────────────────


def run_per_horizon(
    farm: str,
    model_key: str,
    period_name: str,
    training_period: tuple[str, str],
    forecast_period: tuple[str, str],
    n_jobs: int,
):
    """Train and forecast using PerHorizonRunner (ML models)."""
    data_dir = DATA_ROOT / farm / "per_horizon" / "farm_level"
    save_dir = RESULT_ROOT / farm / model_key / period_name
    cfg = MODEL_CONFIGS[model_key]

    runner = PerHorizonRunner(
        data_dir=data_dir,
        registry_key=cfg["registry_key"],
        y_col=ML_Y_COL,
        training_period=training_period,
        forecast_period=forecast_period,
        exog_cols=_resolve_ml_exog_cols(data_dir),
        hyperparameter=dict(cfg["hyperparameter"]),
        dist_name=str(cfg["dist_name"]),
        n_jobs=n_jobs,
        model_name=f"{model_key}_{farm}_{period_name}",
        save_dir=str(save_dir),
    )

    runner.fit()
    result = runner.forecast()
    return result


def run_rolling(
    farm: str,
    model_key: str,
    period_name: str,
    training_period: tuple[str, str],
    forecast_period: tuple[str, str],
):
    """Train and forecast using RollingRunner (deep models)."""
    from src.core.registry import MODEL_REGISTRY

    continuous_dir = DATA_ROOT / farm / "continuous"
    save_dir = RESULT_ROOT / farm / model_key / period_name
    cfg = MODEL_CONFIGS[model_key]

    df = pd.read_csv(
        continuous_dir / "farm_level.csv",
        index_col="basis_time",
        parse_dates=True,
    )

    futr_cols, hist_cols = _resolve_deep_cols(continuous_dir)

    # Train on training period
    train_df = df.loc[training_period[0]:training_period[1]]

    model_cls = MODEL_REGISTRY.get(cfg["registry_key"])
    model = model_cls(
        hyperparameter=dict(cfg["hyperparameter"]),
        model_name=f"{model_key}_{farm}_{period_name}",
    )
    model.fit(
        dataset=train_df,
        y_col=DEEP_Y_COL,
        futr_cols=futr_cols,
        hist_cols=hist_cols,
    )

    runner = RollingRunner(
        model=model,
        dataset=df,
        y_col=DEEP_Y_COL,
        forecast_period=forecast_period,
        futr_cols=futr_cols,
        hist_cols=hist_cols,
    )

    result = runner.run(horizon=DEEP_HORIZON)

    # Save result manually (RollingRunner save_dir triggers model serialization
    # which is broken in NeuralForecast 3.x due to deepcopy issue)
    result_path = save_dir / "forecast_result"
    result_path.mkdir(parents=True, exist_ok=True)
    _save_forecast_result(result, result_path)

    return result


def run_single(
    farm: str,
    model_key: str,
    period_name: str,
    training_period: tuple[str, str],
    forecast_period: tuple[str, str],
    n_jobs: int,
) -> None:
    """Dispatch to the appropriate runner based on model config."""
    cfg = MODEL_CONFIGS[model_key]
    save_dir = RESULT_ROOT / farm / model_key / period_name
    if (save_dir / "forecast_result" / "metadata.yaml").exists():
        print(f"⏭ {farm} / {model_key} / {period_name}  (already done)")
        return
    print(f"▶ {farm} / {model_key} / {period_name}")

    t0 = time.perf_counter()

    if cfg["runner"] == "per_horizon":
        result = run_per_horizon(
            farm, model_key, period_name,
            training_period, forecast_period, n_jobs,
        )
    else:
        result = run_rolling(
            farm, model_key, period_name,
            training_period, forecast_period,
        )

    elapsed = time.perf_counter() - t0
    N = result.basis_index.shape[0]
    H = result.horizon
    save_dir = RESULT_ROOT / farm / model_key / period_name
    print(f"  ✓ ({N}, {H})  {elapsed:.1f}s  → {save_dir}")


# ── plan display ──────────────────────────────────────────────────────


def _print_plan(
    farms: list[str],
    models: list[str],
    period_settings: dict,
) -> None:
    """Print execution plan for user confirmation."""
    separator = "=" * 60

    # ── endog / exog (ML) ─────────────────────────────────────────
    ml_models = [m for m in models if MODEL_CONFIGS[m]["runner"] == "per_horizon"]
    deep_models = [m for m in models if MODEL_CONFIGS[m]["runner"] == "rolling"]

    if ml_models:
        print(separator)
        print(f"  [ML models] Target: {ML_Y_COL}")
        print(separator)

        for farm in farms:
            data_dir = DATA_ROOT / farm / "per_horizon" / "farm_level"
            exog_cols = _resolve_ml_exog_cols(data_dir)
            print(f"\n  [{farm}] exog variables ({len(exog_cols)}):")
            for col in exog_cols:
                print(f"    - {col}")

    # ── endog / exog (Deep) ───────────────────────────────────────
    if deep_models:
        print(f"\n{separator}")
        print(f"  [Deep models] Target: {DEEP_Y_COL}")
        print(separator)

        for farm in farms:
            continuous_dir = DATA_ROOT / farm / "continuous"
            futr_cols, hist_cols = _resolve_deep_cols(continuous_dir)
            print(f"\n  [{farm}] futr_cols ({len(futr_cols)}):")
            for col in futr_cols:
                print(f"    - {col}")
            print(f"  [{farm}] hist_cols ({len(hist_cols)}):")
            for col in hist_cols:
                print(f"    - {col}")

    # ── periods ───────────────────────────────────────────────────
    print(f"\n{separator}")
    print("  Periods")
    print(separator)

    for farm in farms:
        periods = period_settings[farm]
        print(f"\n  [{farm}]")
        for period_name, (train_p, forecast_p) in periods.items():
            print(f"    {period_name}:")
            print(f"      train    : {train_p[0]}  ~  {train_p[1]}")
            print(f"      forecast : {forecast_p[0]}  ~  {forecast_p[1]}")

    # ── models ────────────────────────────────────────────────────
    print(f"\n{separator}")
    print("  Models")
    print(separator)

    for m in models:
        cfg = MODEL_CONFIGS[m]
        hp = cfg["hyperparameter"]
        hp_str = ", ".join(f"{k}={v}" for k, v in hp.items()) if hp else "(defaults)"
        runner_type = cfg["runner"]
        print(f"    {m}  [{cfg['registry_key']}]  runner={runner_type}  {hp_str}")

    # ── summary ───────────────────────────────────────────────────
    n_periods = sum(len(period_settings[f]) for f in farms)
    total = n_periods * len(models)
    print(f"\n{separator}")
    print(f"  Total runs: {len(farms)} farms × {len(models)} models = {total} runs")
    print(separator)


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI arguments and run all farm/model/period combinations."""
    parser = argparse.ArgumentParser(description="Run ML / Deep model forecasts")
    parser.add_argument(
        "--farms",
        nargs="+",
        default=None,
        help="Farm names (default: all in training_data_new)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=8, help="Parallel horizon jobs (default: 8)"
    )
    args = parser.parse_args()

    with open(PERIOD_JSON) as f:
        period_settings = json.load(f)

    farms = args.farms or sorted(period_settings.keys())
    farms = [f for f in farms if f in period_settings]

    _print_plan(farms, args.models, period_settings)

    confirm = input("\nProceed? [Y/n] ").strip().lower()
    if confirm not in ("y", "yes", ""):
        print("Aborted.")
        return

    print()
    for farm in farms:
        periods = period_settings[farm]
        for period_name, (training_period, forecast_period) in periods.items():
            for model_key in args.models:
                run_single(
                    farm=farm,
                    model_key=model_key,
                    period_name=period_name,
                    training_period=tuple(training_period),
                    forecast_period=tuple(forecast_period),
                    n_jobs=args.n_jobs,
                )

    print("All done.")


if __name__ == "__main__":
    main()
