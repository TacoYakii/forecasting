import pandas as pd
from typing import List
from ..config import NWPSourceConfig

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
