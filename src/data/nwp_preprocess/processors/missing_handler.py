"""
Missing data detection and interpolation for NWP data.

Consolidates missing data handling from legacy missing.py (KMA) and
find_replacement_file (ECMWF) into a unified handler.

Strategy: for each missing value, search backwards in time (same hour)
for the nearest valid data point, with no upper limit on search range.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MissingHandler:
    """
    Handles missing NWP data by replacing with the nearest valid past data.

    For each missing (NaN) value, searches backwards day-by-day at the same
    forecast hour until a valid value is found. No upper limit on search
    range — continues until data is found or no more past files exist.

    Replaced values have their is_valid flag set to False to indicate
    they are not original observations.
    """

    def fill(
        self,
        df: pd.DataFrame,
        coord_key: str,
        output_dir: Path,
        basis_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Fill missing values by searching nearest past valid data.

        Looks for previously processed CSV files in output_dir/coord_key/
        at the same hour, going back day by day until valid data is found.

        Args:
            df: DataFrame with potential NaN values.
            coord_key: Coordinate identifier (folder name for past data).
            output_dir: Root output directory where past CSVs are stored.
            basis_time: The basis time of the current file.

        Returns:
            DataFrame with NaN values filled where possible.
            Filled rows have is_valid = False.
        """
        if not df.isna().any().any():
            return df

        df = df.copy()
        data_cols = [
            c for c in df.columns
            if c not in {"basis_time", "is_valid", "pressure_level"}
        ]

        # Find rows with any missing data
        missing_mask = df[data_cols].isna().any(axis=1)
        if not missing_mask.any():
            return df

        missing_count = missing_mask.sum()
        logger.info(
            f"[{coord_key}] {basis_time}: {missing_count} rows with missing data, "
            f"searching past files..."
        )

        past_dir = output_dir / coord_key
        if not past_dir.exists():
            logger.warning(
                f"No past data directory for {coord_key}. "
                f"Cannot fill {missing_count} missing rows."
            )
            return df

        # Search backwards day by day
        days_back = 1
        filled_indices = set()
        remaining_missing = set(df.index[missing_mask])

        while remaining_missing:
            past_basis_time = basis_time - pd.Timedelta(days=days_back)
            past_file = self._find_past_file(
                past_dir, past_basis_time, current_basis_time=basis_time,
            )

            if past_file is None:
                # No more past data available
                if days_back > 365:
                    logger.warning(
                        f"[{coord_key}] Searched back {days_back} days, "
                        f"giving up on {len(remaining_missing)} missing rows."
                    )
                    break
                days_back += 1
                continue

            try:
                past_df = pd.read_csv(past_file, index_col=0, parse_dates=True)
            except Exception as e:
                logger.warning(f"Failed to read past file {past_file}: {e}")
                days_back += 1
                continue

            # Try to fill remaining missing values
            newly_filled = []
            for idx in list(remaining_missing):
                # Compute corresponding index in past data
                past_idx = idx - pd.Timedelta(days=days_back)

                if past_idx not in past_df.index:
                    continue

                past_row = past_df.loc[past_idx]

                # Skip rows that were themselves filled (is_valid=False)
                # to avoid chain-propagation of stale data.
                if "is_valid" in past_row.index and not past_row["is_valid"]:
                    continue

                # Check if past row has the data we need
                missing_cols_for_row = df.loc[idx, data_cols][
                    df.loc[idx, data_cols].isna()
                ].index.tolist()

                if all(
                    col in past_row.index and pd.notna(past_row[col])
                    for col in missing_cols_for_row
                ):
                    for col in missing_cols_for_row:
                        df.loc[idx, col] = past_row[col]
                    df.loc[idx, "is_valid"] = False
                    newly_filled.append(idx)

            for idx in newly_filled:
                remaining_missing.discard(idx)
                filled_indices.add(idx)

            days_back += 1

        if filled_indices:
            logger.info(
                f"[{coord_key}] Filled {len(filled_indices)} rows from past data"
            )
        if remaining_missing:
            logger.warning(
                f"[{coord_key}] Could not fill {len(remaining_missing)} rows"
            )

        return df

    @staticmethod
    def _find_past_file(
        coord_dir: Path,
        past_basis_time: pd.Timestamp,
        current_basis_time: Optional[pd.Timestamp] = None,
    ) -> Optional[Path]:
        """Find a past CSV file for the given basis time.

        Tries both YYYY-MM-DD_HH and YYYYMMDDHH filename formats.
        Skips files with basis_time >= current_basis_time to avoid
        reading output from the current parallel batch.
        """
        if (
            current_basis_time is not None
            and past_basis_time >= current_basis_time
        ):
            return None
        for fmt in ["%Y-%m-%d_%H", "%Y%m%d%H"]:
            candidate = coord_dir / f"{past_basis_time.strftime(fmt)}.csv"
            if candidate.exists():
                return candidate
        return None
