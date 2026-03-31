"""Shared test fixtures: synthetic wind power data in CSV format."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _generate_synthetic_wind_power(
    n_hours: int = 720,
    start: str = "2023-01-01",
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hourly wind power data with correlated features.

    Mimics real-world CSV structure: DatetimeIndex, power (target),
    wind_speed / temperature (features).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    # Wind speed: diurnal cycle + noise
    wind_speed = 7.0 + 3.0 * np.sin(2 * np.pi * t / 24 + np.pi / 4) + rng.normal(0, 1.0, n_hours)
    wind_speed = np.maximum(wind_speed, 0.5)

    # Temperature: slow seasonal + diurnal
    temperature = 15.0 + 5.0 * np.sin(2 * np.pi * t / (24 * 30)) + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1.5, n_hours)

    # Power: cubic-ish relationship with wind + noise, clipped to [0, 100]
    power = 5.0 + 1.2 * wind_speed**1.5 - 0.3 * temperature + rng.normal(0, 3.0, n_hours)
    power = np.clip(power, 0.5, 100.0)

    index = pd.date_range(start, periods=n_hours, freq=freq)
    return pd.DataFrame(
        {"power": power, "wind_speed": wind_speed, "temperature": temperature},
        index=index,
    )


# ---------------------------------------------------------------------------
# Fixtures — continuous time series CSV
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> str:
    """Write a single continuous CSV and return its path."""
    path = tmp_path_factory.mktemp("data") / "wind_power.csv"
    df = _generate_synthetic_wind_power(n_hours=720)
    df.to_csv(path, index_label="datetime")
    return str(path)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Return the synthetic DataFrame directly (for tests that don't need CSV)."""
    return _generate_synthetic_wind_power(n_hours=720)


# ---------------------------------------------------------------------------
# Fixtures — per-horizon CSVs (for PerHorizonRunner)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def per_horizon_csv_dir(tmp_path_factory) -> str:
    """Generate horizon_1.csv … horizon_3.csv and return the directory path.

    Each CSV has `basis_time` as the index column, matching the format
    that PerHorizonRunner._load_dataset() expects.
    """
    data_dir = tmp_path_factory.mktemp("per_horizon")
    rng = np.random.default_rng(99)
    n_rows = 720
    index = pd.date_range("2023-01-01", periods=n_rows, freq="h", name="basis_time")

    for h in range(1, 4):
        wind_speed = 7.0 + rng.normal(0, 2.0, n_rows)
        temperature = 15.0 + rng.normal(0, 3.0, n_rows)
        # Power shifted by horizon to simulate lead-time effect
        power = 5.0 + 1.2 * wind_speed**1.5 - 0.3 * temperature + rng.normal(0, 2.0 + h, n_rows)
        power = np.clip(power, 0.5, 100.0)

        df = pd.DataFrame(
            {"power": power, "wind_speed": wind_speed, "temperature": temperature},
            index=index,
        )
        df.to_csv(data_dir / f"horizon_{h}.csv", index_label="basis_time")

    return str(data_dir)
