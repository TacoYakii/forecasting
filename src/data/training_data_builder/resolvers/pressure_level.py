"""
Pressure-level NWP resolver for KMA-style data.

For each forecast-time row, selects the pressure level whose geometric
height is closest to the turbine's hub height.

Expected directory structure::

    root/
    ├── A/                    # data-point group (from TurbineInfo.nwp_group)
    │   ├── 85000/            # pressure level
    │   │   ├── 2024-01-01_00.csv
    │   │   └── ...
    │   └── 92500/
    │       └── ...
    └── B/
        └── ...
"""
from pathlib import Path
from typing import Dict

import pandas as pd

from ..config import TurbineInfo
from .base import AbstractNWPResolver


class PressureLevelResolver(AbstractNWPResolver):
    """Select NWP data from the pressure level closest to hub height.

    Each turbine is assigned a KMA data-point group via
    ``TurbineInfo.nwp_group`` (e.g. ``"A"``, ``"B"``).  Within that
    group directory, pressure-level subdirectories (``"85000"``,
    ``"92500"``, …) contain per-basis-time CSVs with a ``height``
    column (geometric height in metres).

    Args:
        turbine_info: Mapping of turbine IDs to their metadata.
        height_column: Column containing geometric height.
        drop_columns: Columns to drop from the result (e.g. intermediate
            height-related columns).
    """

    def __init__(
        self,
        turbine_info: Dict[str, TurbineInfo],
        height_column: str = "height",
        drop_columns: list[str] | None = None,
    ):
        self.turbine_info = turbine_info
        self.height_column = height_column
        self.drop_columns = drop_columns or ["height"]

    def get_data_path(self, nwp_root: Path, turbine_id: str) -> Path:
        """Return ``nwp_root / {group}`` for the turbine's KMA group."""
        info = self.turbine_info.get(turbine_id)
        if info is None or info.nwp_group is None:
            raise ValueError(
                f"Turbine '{turbine_id}' has no nwp_group assigned. "
                f"PressureLevelResolver requires TurbineInfo.nwp_group."
            )
        group_dir = nwp_root / info.nwp_group
        if not group_dir.exists():
            raise FileNotFoundError(
                f"KMA group directory not found: {group_dir}"
            )
        return group_dir

    def load_basis_time(
        self,
        data_path: Path,
        basis_time_str: str,
        turbine_id: str,
    ) -> pd.DataFrame:
        """Load all pressure levels and select the closest to hub height
        for each forecast-time row.

        Args:
            data_path: Group directory (``nwp_root/{group}``).
            basis_time_str: NWP basis time file stem (e.g. ``"2024-01-01_00"``).
            turbine_id: Turbine identifier.

        Steps:
            1. Discover pressure-level subdirectories under the group.
            2. Load each level's CSV for the given basis_time.
            3. Build a height matrix (rows = forecast_time, cols = levels).
            4. For each row, pick the level minimising |height − hub_height|.
            5. Assemble the result from the best-level rows.
        """
        hub_height = self.turbine_info[turbine_id].hub_height

        # Discover pressure levels within the group directory
        level_dirs = sorted(
            d.name for d in data_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if not level_dirs:
            raise FileNotFoundError(
                f"No pressure-level subdirectories found in {data_path}"
            )

        # Load each level
        level_dfs: Dict[str, pd.DataFrame] = {}
        for lv in level_dirs:
            csv_path = data_path / lv / f"{basis_time_str}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = "forecast_time"
            level_dfs[lv] = df

        if not level_dfs:
            raise FileNotFoundError(
                f"No CSVs found for basis_time {basis_time_str} in {data_path}"
            )

        # Build height matrix and find best level per row
        heights = pd.DataFrame(
            {lv: df[self.height_column] for lv, df in level_dfs.items()}
        )
        best_level = (heights - hub_height).abs().idxmin(axis=1)

        # Assemble result: select all rows for a winning level at once
        result_rows = []
        for lv, group in best_level.groupby(best_level):
            result_rows.append(level_dfs[lv].loc[group.index])
        
        result = pd.concat(result_rows).sort_index()

        # Drop height-related columns
        cols_to_drop = [c for c in self.drop_columns if c in result.columns]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result
