"""Model-test fixtures: train/test splits, common constants."""

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants shared across model tests
# ---------------------------------------------------------------------------

Y_COL = "power"
EXOG_COLS = ["wind_speed", "temperature"]

# Time splits (within the 720-hour synthetic data: 2023-01-01 ~ 2023-01-30)
TRAIN_END = "2023-01-20"
FORECAST_START = "2023-01-21"
FORECAST_END = "2023-01-25"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def train_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Training slice of the synthetic data."""
    return synthetic_df.loc[:TRAIN_END].copy()


@pytest.fixture()
def forecast_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Forecast-period slice of the synthetic data."""
    return synthetic_df.loc[FORECAST_START:FORECAST_END].copy()


@pytest.fixture()
def full_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Full dataset (train + forecast) for RollingRunner."""
    return synthetic_df.loc[:FORECAST_END].copy()
