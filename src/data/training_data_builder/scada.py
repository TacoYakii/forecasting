"""
SCADA data loading and feature engineering utilities.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_scada(
    path: Path,
    time_column: str = "basis_time",
) -> pd.DataFrame:
    """Load a SCADA CSV and return a DataFrame with a DatetimeIndex.

    Args:
        path: Path to the SCADA CSV file.
        time_column: Name of the column to use as index.

    Returns:
        DataFrame indexed by ``time_column`` (parsed as datetime).
    """
    df = pd.read_csv(path)
    if time_column not in df.columns:
        raise KeyError(
            f"'{time_column}' column not found in {path}.  "
            f"Please check the SCADA CSV."
        )
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.set_index(time_column)
    return df


def create_lagged_features(
    scada: pd.DataFrame,
    max_lag: int = 6,
) -> pd.DataFrame:
    """Create lagged SCADA features and propagate validity flags.

    For each lag ``t`` in ``[0, max_lag]``, every data column is shifted
    by ``t`` rows and renamed to ``basis_time_lag{t}_{col}``.  The
    ``is_valid`` flag is the ``AND`` of all individual lag validities.

    Args:
        scada: SCADA DataFrame (output of :func:`load_scada`).
        max_lag: Maximum number of lag hours.

    Returns:
        DataFrame with lagged features and a single ``is_valid`` column.
    """
    is_valid = scada["is_valid"].copy() if "is_valid" in scada.columns else None
    data = scada.drop(columns=["is_valid"], errors="ignore")

    lagged_dfs: list[pd.DataFrame] = []
    lagged_valids: list[pd.Series] = []

    for lag in range(max_lag + 1):
        shifted = data.shift(lag)
        shifted = shifted.rename(
            columns={col: f"basis_time_lag{lag}_{col}" for col in shifted.columns}
        )
        lagged_dfs.append(shifted)
        if is_valid is not None:
            lagged_valids.append(is_valid.shift(lag))

    combined = pd.concat(lagged_dfs, axis=1)
    combined.dropna(inplace=True)

    if lagged_valids:
        combined_valid = pd.concat(lagged_valids, axis=1).reindex(combined.index)
        combined["is_valid"] = combined_valid.all(axis=1)

    return combined


def prepare_target(
    scada: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Prepare SCADA target columns for a specific forecast horizon.

    Shifts SCADA data backward by *horizon* rows so that when aligned
    with basis_time, the values represent the observation at
    ``forecast_time = basis_time + horizon``.

    Columns are prefixed with ``forecast_time_`` to distinguish them
    from lag features.

    Args:
        scada: SCADA DataFrame.
        horizon: Forecast horizon in hours.

    Returns:
        DataFrame with ``forecast_time_{col}`` columns and ``is_valid``.
    """
    is_valid = scada["is_valid"].copy() if "is_valid" in scada.columns else None
    data = scada.drop(columns=["is_valid"], errors="ignore")

    if horizon > 0:
        # Shift backward: row at index t now contains the value at t + horizon
        target = data.shift(-horizon)
    else:
        target = data.copy()

    target = target.rename(
        columns={col: f"forecast_time_{col}" for col in target.columns}
    )

    if is_valid is not None:
        if horizon > 0:
            target["target_is_valid"] = is_valid.shift(-horizon)
        else:
            target["target_is_valid"] = is_valid

    target.dropna(inplace=True)
    return target


def clip_positive_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Clip columns whose base name matches *columns* to be non-negative.

    Matching is by suffix: specifying ``["KPX_pwr"]`` will clip any
    column that ends with ``KPX_pwr``, such as ``forecast_time_KPX_pwr``
    or ``basis_time_lag0_KPX_pwr``.
    """
    df_copy = df.copy()
    to_clip = [
        col for col in df_copy.columns
        if any(col.endswith(base) for base in columns)
    ]
    if to_clip:
        df_copy[to_clip] = df_copy[to_clip].clip(lower=0)
    return df_copy
