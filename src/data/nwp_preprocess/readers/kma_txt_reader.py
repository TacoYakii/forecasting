"""
KMA (Korea Meteorological Administration) TXT file reader.

Reads KMA's point-based text format, performs pressure level separation
and variable pivoting, producing the same Dict[str, DataFrame] interface
as the Grib2Reader.

This consolidates the logic from the legacy modules:
    - preprocess/KMA/txt_to_csv.py   (TXT parsing)
    - preprocess/KMA/pressure_level.py (pressure level separation)
    - preprocess/KMA/category.py      (variable pivoting by forecast time)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .base import AbstractReader

logger = logging.getLogger(__name__)

# KMA variable code → readable name mapping
DEFAULT_PARAMETER_CODES: Dict[int, str] = {
    2009: "vertical_velocity",
    2002: "U-component",
    2003: "V-component",
    3005: "geo_potential_height",
    0: "temperature",
    1001: "humidity",
    1194: "RH_wrt_ICE_ON_PLEV",
}

# Standard column names for KMA TXT files
TXT_COLUMNS = ["basis_time", "forecast_time", "data_type", "pressure_level", "value"]


class KMATxtReader(AbstractReader):
    """
    Reader for KMA text-format NWP files.

    KMA data is already point-specific (requested by lat/lon),
    so each file contains data for a single location. The reader
    parses the raw text, separates by pressure level, and pivots
    variables into columns.

    Args:
        parameter_codes: Mapping of KMA variable codes to readable names.
            Defaults to the standard KMA parameter set.
        point_id: Identifier for this point (used as dict key in output).
            If None, derived from parent directory name.
    """

    def __init__(
        self,
        parameter_codes: Optional[Dict[int, str]] = None,
        point_id: Optional[str] = None,
    ):
        self.parameter_codes = parameter_codes or DEFAULT_PARAMETER_CODES
        self.point_id = point_id

    def validate_file(self, file_path: Path) -> bool:
        """Check TXT file validity.

        Validates:
        1. File exists
        2. File is not empty
        3. File contains parseable data lines
        """
        try:
            if not file_path.exists():
                return False

            if file_path.stat().st_size == 0:
                logger.warning(f"Empty file: {file_path}")
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                logger.warning(f"No data lines in: {file_path}")
                return False

            return True

        except Exception as e:
            logger.warning(f"File validation failed for {file_path}: {e}")
            return False

    def list_files(self, input_dir: Path) -> List[Path]:
        """Return sorted list of TXT files in the directory."""
        files = list(input_dir.glob("*.txt"))
        return sorted(files, key=lambda p: p.stem)

    def _parse_txt(self, file_path: Path) -> pd.DataFrame:
        """Parse a raw KMA TXT file into a DataFrame.

        Args:
            file_path: Path to the TXT file.

        Returns:
            DataFrame with columns: basis_time, forecast_time,
            data_type, pressure_level, value
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        data = [line.split() for line in lines if line.strip()]

        if not data:
            raise ValueError(f"No valid data in {file_path}")

        df = pd.DataFrame(data, columns=TXT_COLUMNS)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["data_type"] = pd.to_numeric(df["data_type"], errors="coerce").astype(int)
        df["pressure_level"] = pd.to_numeric(df["pressure_level"], errors="coerce").astype(int)

        return df

    def _pivot_by_pressure_and_variable(
        self, raw_df: pd.DataFrame, basis_time: pd.Timestamp, point_id: str,
    ) -> Dict[str, pd.DataFrame]:
        """Pivot raw parsed data into per-pressure-level DataFrames.

        Groups by pressure level, then pivots data_type codes into columns
        with readable names.

        Args:
            raw_df: Raw parsed DataFrame from _parse_txt.
            basis_time: The basis (analysis) time for this file.
            point_id: Identifier for this point (used as dict key prefix).

        Returns:
            Dict mapping "point_id/pressure_level" to pivoted DataFrames.
        """
        raw_df["forecast_time"] = pd.to_datetime(raw_df["forecast_time"], format="%Y%m%d%H")

        result: Dict[str, pd.DataFrame] = {}

        for pressure_level, pressure_group in raw_df.groupby("pressure_level"):
            # Pivot: one column per variable
            pivoted = pressure_group.pivot_table(
                index="forecast_time",
                columns="data_type",
                values="value",
                aggfunc="first",
            )

            # Rename columns using parameter codes
            col_mapping = {
                col: self.parameter_codes.get(int(col), f"var_{int(col)}")
                for col in pivoted.columns
            }
            pivoted.rename(columns=col_mapping, inplace=True)

            pivoted.sort_index(inplace=True)
            pivoted["basis_time"] = basis_time
            pivoted["pressure_level"] = pressure_level
            pivoted["is_valid"] = True

            coord_key = f"{point_id}/{pressure_level}"
            result[coord_key] = pivoted

        return result

    def read(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Read a KMA TXT file and return per-coordinate DataFrames.

        Since KMA data is point-specific, the result typically contains
        entries keyed by "point_id/pressure_level".

        Args:
            file_path: Path to the KMA TXT file.

        Returns:
            Dict mapping coordinate keys to DataFrames.
            Each DataFrame has:
                - index: forecast_time (DatetimeIndex)
                - columns: meteorological variables + "basis_time" + "pressure_level" + "is_valid"
        """
        # Resolve point_id without mutating instance state
        point_id = self.point_id or file_path.parent.name

        # Extract basis_time from filename (format: YYYY-MM-DD_HH or YYYYMMDDHH)
        stem = file_path.stem
        try:
            basis_time = pd.to_datetime(stem, format="%Y-%m-%d_%H")
        except ValueError:
            basis_time = pd.to_datetime(stem, format="%Y%m%d%H")

        # Parse and pivot
        raw_df = self._parse_txt(file_path)
        result = self._pivot_by_pressure_and_variable(raw_df, basis_time, point_id)

        return result
