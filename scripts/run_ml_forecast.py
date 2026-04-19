"""Run ML (per-horizon) model forecasts for all farms and CV periods.

Usage:
    uv run python scripts/run_ml_forecast.py
    uv run python scripts/run_ml_forecast.py --farms sinan dongbok
    uv run python scripts/run_ml_forecast.py --models ngboost_normal catboost
    uv run python scripts/run_ml_forecast.py --n-jobs 8

Deep learning models (TFT, NHITS, DeepAR) are in separate scripts:
    - scripts/run_tft_forecast.py
    - scripts/run_nhits_forecast.py
    - scripts/run_deepar_forecast.py
"""

import argparse
import json
import re
import warnings

warnings.filterwarnings("ignore")
import time
from pathlib import Path

import pandas as pd

import src.models.machine_learning  # noqa: F401  (register ML models)
from src.core.runner import PerHorizonRunner

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

# ── model configs ─────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "ngboost_normal": {
        "registry_key": "ngboost",
        "hyperparameter": {"Dist": "normal"},
        "dist_name": "normal",
    },
    "catboost": {
        "registry_key": "catboost",
        "hyperparameter": {
            "loss_function": "RMSEWithUncertainty",
            "posterior_sampling": True,
            "silent": True,
            "allow_writing_files": False,
            "thread_count": 1,  # avoid contention with joblib n_jobs workers
        },
        "dist_name": "normal",
    },
    "pgbm": {
        "registry_key": "pgbm",
        "hyperparameter": {
            "Dist": "normal",
            "device": "cpu",  # CPU only — GPU is used by parallel deep models
        },
        "dist_name": "normal",
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


def run_single(
    farm: str,
    model_key: str,
    period_name: str,
    training_period: tuple[str, str],
    forecast_period: tuple[str, str],
    n_jobs: int,
) -> None:
    """Run a single per-horizon ML model for one (farm, period)."""
    save_dir = RESULT_ROOT / farm / model_key / period_name
    if (save_dir / "forecast_result" / "metadata.yaml").exists():
        print(f"⏭ {farm} / {model_key} / {period_name}  (already done)")
        return
    print(f"▶ {farm} / {model_key} / {period_name}")

    t0 = time.perf_counter()
    result = run_per_horizon(
        farm, model_key, period_name,
        training_period, forecast_period, n_jobs,
    )
    elapsed = time.perf_counter() - t0
    N = result.basis_index.shape[0]
    H = result.horizon
    print(f"  ✓ ({N}, {H})  {elapsed:.1f}s  → {save_dir}")


# ── plan display ──────────────────────────────────────────────────────


def _print_plan(
    farms: list[str],
    models: list[str],
    period_settings: dict,
) -> None:
    """Print execution plan for user confirmation."""
    separator = "=" * 60

    print(separator)
    print(f"  [ML models] Target: {ML_Y_COL}")
    print(separator)

    for farm in farms:
        data_dir = DATA_ROOT / farm / "per_horizon" / "farm_level"
        exog_cols = _resolve_ml_exog_cols(data_dir)
        print(f"\n  [{farm}] exog variables ({len(exog_cols)}):")
        for col in exog_cols:
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
        print(f"    {m}  [{cfg['registry_key']}]  {hp_str}")

    # ── summary ───────────────────────────────────────────────────
    n_periods = sum(len(period_settings[f]) for f in farms)
    total = n_periods * len(models)
    print(f"\n{separator}")
    print(f"  Total runs: {len(farms)} farms × {len(models)} models = {total} runs")
    print(separator)


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI arguments and run all farm/model/period combinations."""
    parser = argparse.ArgumentParser(description="Run per-horizon ML forecasts")
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
