"""
Data validation and is_valid flag management for NWP data.

Consolidates integrity checks from legacy is_valid.py and
check_grib_file_integrity into a unified validator.
"""
import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates NWP DataFrames and manages the is_valid flag.

    Checks for:
    - NaN values in data columns
    - Values outside physically reasonable ranges (optional)

    The is_valid column is set to False for any row that has
    NaN values or range violations.
    """

    # Default physical range limits using column name prefixes.
    # Matches VariableDeriver output: wspd10, wdir10, wspd100, wdir100, etc.
    DEFAULT_PREFIX_RANGES: Dict[str, tuple] = {
        "wspd": (0, 120),           # m/s — matches wspd10, wspd100, ...
        "wdir": (0, 360),           # degrees — matches wdir10, wdir100, ...
        "temperature": (150, 350),  # Kelvin
    }

    def __init__(
        self,
        range_limits: Optional[Dict[str, tuple]] = None,
        exclude_cols: Optional[List[str]] = None,
    ):
        """
        Args:
            range_limits: Optional dict of {prefix: (min, max)} for range checks.
                Keys are treated as column name prefixes.
                If None, uses DEFAULT_PREFIX_RANGES for columns that match.
            exclude_cols: Columns to exclude from NaN checking
                (e.g., metadata columns).
        """
        self.range_limits = range_limits or self.DEFAULT_PREFIX_RANGES
        self.exclude_cols = set(exclude_cols or ["basis_time", "is_valid", "pressure_level"])

    def _match_columns_by_prefix(
        self, columns: List[str], prefix: str
    ) -> List[str]:
        """Return columns that start with the given prefix."""
        return [c for c in columns if c.startswith(prefix)]

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a DataFrame and set is_valid flags.

        Rows with any NaN in data columns or values outside range limits
        will have is_valid = False. Range limits are matched by column
        name prefix (e.g., prefix 'wspd' matches 'wspd10', 'wspd100').

        Args:
            df: Input DataFrame. May or may not already have an is_valid column.

        Returns:
            DataFrame with is_valid column updated.
        """
        df = df.copy()

        # Initialize is_valid if not present
        if "is_valid" not in df.columns:
            df["is_valid"] = True

        # Check for NaN values in data columns
        data_cols = [c for c in df.columns if c not in self.exclude_cols]
        nan_mask = df[data_cols].isna().any(axis=1)

        if nan_mask.any():
            nan_count = nan_mask.sum()
            total_count = len(df)
            logger.info(
                "%d/%d (%.1f%%) rows with NaN values",
                nan_count, total_count, nan_count / total_count * 100,
            )
            df.loc[nan_mask, "is_valid"] = False

        # Range validation (prefix-based matching)
        for prefix, (min_val, max_val) in self.range_limits.items():
            matched_cols = self._match_columns_by_prefix(data_cols, prefix)
            for col in matched_cols:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                if out_of_range.any():
                    count = out_of_range.sum()
                    logger.warning(
                        "Column '%s': %d values outside range [%s, %s]",
                        col, count, min_val, max_val,
                    )
                    df.loc[out_of_range, "is_valid"] = False

        return df
