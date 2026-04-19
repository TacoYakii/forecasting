import logging
import re
from typing import List

import numpy as np
import pandas as pd

from ..config import NWPSourceConfig

logger = logging.getLogger(__name__)


def recompute_wind_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute wdir from averaged U/V components.

    After averaging (farm-level or temporal hierarchy), wind direction
    columns are physically incorrect because angles were averaged as
    scalars.  This function recomputes wdir from the (correctly
    averaged) U/V vector components.  Wind speed (wspd) is left as the
    scalar average since ``avg(||v||)`` is the desired quantity.

    Handles three naming conventions (case-insensitive for U/V lookup):
        - ECMWF-style:    ``{prefix}_forecast_wdir{height}`` from ``u{height}/v{height}``
        - KMA-style:      ``{prefix}_forecast_wdir`` from ``U-component/V-component``
        - Observed-style: ``{prefix}_observed_wdir`` from ``{prefix}_observed_u/v``

    Args:
        df: DataFrame with wind U/V and derived wdir columns.

    Returns:
        DataFrame with wdir recomputed in-place.

    Example:
        >>> df = recompute_wind_derived(averaged_df)
    """
    CALM_THRESHOLD = 0.01

    cols = df.columns.tolist()
    col_lower = {c.lower(): c for c in cols}

    def _find_col(name: str):
        return col_lower.get(name.lower())

    def _recompute_wdir(u_col, v_col, wdir_col):
        u = df[u_col].astype(float)
        v = df[v_col].astype(float)
        wspd = np.sqrt(u**2 + v**2)
        wdir = (180 + np.degrees(np.arctan2(u, v))) % 360
        df[wdir_col] = np.where(wspd < CALM_THRESHOLD, 0.0, wdir)

    recomputed = set()

    # ECMWF-style: {prefix}_forecast_wdir{height} from u{height}/v{height}
    for col in cols:
        m = re.match(r"(.+_forecast_)wdir(\d+)$", col, re.IGNORECASE)
        if not m:
            continue
        prefix, height = m.group(1), m.group(2)
        u_col = _find_col(f"{prefix}u{height}")
        v_col = _find_col(f"{prefix}v{height}")
        if u_col and v_col:
            _recompute_wdir(u_col, v_col, col)
            recomputed.add(col)

    # KMA-style: {prefix}_forecast_wdir from U-component/V-component
    for col in cols:
        if col in recomputed:
            continue
        m = re.match(r"(.+_forecast_)wdir$", col, re.IGNORECASE)
        if not m:
            continue
        prefix = m.group(1)
        u_col = _find_col(f"{prefix}U-component")
        v_col = _find_col(f"{prefix}V-component")
        if u_col and v_col:
            _recompute_wdir(u_col, v_col, col)
            recomputed.add(col)

    # Observed-style: {prefix}observed_wdir from {prefix}observed_u/v
    # Matches both "forecast_time_observed_wdir" and bare "observed_wdir"
    for col in cols:
        if col in recomputed:
            continue
        m = re.match(r"(.*observed_)wdir$", col, re.IGNORECASE)
        if not m:
            continue
        prefix = m.group(1)
        u_col = _find_col(f"{prefix}u")
        v_col = _find_col(f"{prefix}v")
        if u_col and v_col:
            _recompute_wdir(u_col, v_col, col)
            recomputed.add(col)

    all_wdir = [
        c for c in cols
        if re.search(r"(forecast_|observed_)wdir", c, re.IGNORECASE)
    ]
    unmatched = [c for c in all_wdir if c not in recomputed]
    if unmatched:
        logger.warning(
            "Wind direction columns could not be recomputed (no matching "
            "U/V components found): %s. These remain as scalar averages.",
            unmatched,
        )

    return df


def format_final_dataset(
    merged: pd.DataFrame,
    nwp_sources: List[NWPSourceConfig],
    is_continuous: bool = False,
) -> pd.DataFrame:
    """Consistently reorder columns and handle is_valid flag for training datasets.
    
    Expected order:
     1. SCADA Targets (forecast_time_*)
     2. NWP predictions (grouped by source)
     3. Lagged features (basis_time_lag*)
     4. everything else
     5. is_valid
    """
    # Combine validity
    valid_cols = [c for c in merged.columns if "is_valid" in c]
    if valid_cols:
        merged["is_valid"] = merged[valid_cols].all(axis=1)
        drop_cols = [c for c in valid_cols if c != "is_valid"]
        merged = merged.drop(columns=drop_cols)

    # ---------------------------------------------
    # Reorder columns
    # ---------------------------------------------
    ordered_cols = []
    if "forecast_time" in merged.columns and not is_continuous:
        ordered_cols.append("forecast_time")

    # 1. SCADA Targets
    target_cols = [c for c in merged.columns if c.startswith("forecast_time_") and c != "forecast_time"]
    ordered_cols.extend(sorted(target_cols))

    # 2. NWP predictions
    for src in nwp_sources:
        nwp_cols = [c for c in merged.columns if c.startswith(f"{src.name}_forecast_")]
        ordered_cols.extend(sorted(nwp_cols))

    # 3. Lagged features
    lag_cols = [c for c in merged.columns if c.startswith("basis_time_lag")]
    ordered_cols.extend(sorted(lag_cols))

    # 4. is_valid
    valid_col = ["is_valid"] if "is_valid" in merged.columns else []

    # Any remaining
    remaining = [c for c in merged.columns if c not in ordered_cols and c != "is_valid"]
    ordered_cols.extend(sorted(remaining))
    ordered_cols.extend(valid_col)

    return merged[ordered_cols]
