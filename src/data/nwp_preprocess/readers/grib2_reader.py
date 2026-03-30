"""
GRIB2 file reader for ECMWF, NOAA, and other GRIB2-based NWP sources.

Reads GRIB2 and NetCDF files using xarray, validates file
integrity, and returns per-coordinate DataFrames.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xarray as xr

from .base import AbstractReader

logger = logging.getLogger(__name__)


class Grib2Reader(AbstractReader):
    """
    Reader for GRIB2 and NetCDF format NWP files.

    Supports ECMWF, NOAA/GFS, and any GRIB2-based data source.
    Also reads merged NetCDF (.nc) files produced by concat_grib2_by_time.

    Args:
        cfgrib_kwargs: Optional keyword arguments passed to xr.open_dataset's
            backend_kwargs for cfgrib engine. Useful for filtering by keys, etc.
        min_file_size: Minimum file size in bytes to consider valid (default: 1000).
        extensions: File extensions to search for (default: [".gb2", ".grib2"]).
    """

    def __init__(
        self,
        cfgrib_kwargs: Optional[dict] = None,
        min_file_size: int = 1000,
        extensions: Optional[List[str]] = None,
    ):
        self.cfgrib_kwargs = cfgrib_kwargs or {}
        self.min_file_size = min_file_size
        self.extensions = extensions or [".gb2", ".grib2", ".nc"]

    def _open_dataset(self, file_path: Path) -> xr.Dataset:
        """Open a file with the appropriate engine based on extension."""
        if file_path.suffix == ".nc":
            return xr.open_dataset(file_path, decode_timedelta=True)
        else:
            return xr.open_dataset(
                file_path,
                engine="cfgrib",
                decode_timedelta=True,
                backend_kwargs=self.cfgrib_kwargs,
            )

    def validate_file(self, file_path: Path) -> bool:
        """Check file integrity.

        Validates:
        1. File exists
        2. File size >= min_file_size
        3. File is readable by cfgrib and contains data
        """
        try:
            if not file_path.exists():
                return False

            if file_path.stat().st_size < self.min_file_size:
                logger.warning(f"File too small ({file_path.stat().st_size} bytes): {file_path}")
                return False

            with self._open_dataset(file_path) as ds:
                if len(ds.data_vars) == 0:
                    logger.warning(f"No data variables in: {file_path}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"File validation failed for {file_path}: {e}")
            return False

    def list_files(self, input_dir: Path) -> List[Path]:
        """Return sorted list of GRIB2/NetCDF files in the directory."""
        files = []
        for ext in self.extensions:
            files.extend(input_dir.glob(f"*{ext}"))

        return sorted(files, key=lambda p: p.stem)

    def read(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Read a GRIB2 or NetCDF file and return per-coordinate DataFrames.

        Each unique (latitude, longitude) pair in the GRIB2 file becomes
        a separate entry in the returned dictionary.

        Args:
            file_path: Path to the GRIB2 or NetCDF file.

        Returns:
            Dict mapping "lat_lon" keys to DataFrames.
            Each DataFrame has:
                - index: forecast_time (DatetimeIndex)
                - columns: meteorological variables + "basis_time" + "is_valid"
        """
        ds = self._open_dataset(file_path)

        # Extract basis_time from dataset metadata, filename as fallback
        basis_time = self._extract_basis_time(ds, file_path)

        df = ds.to_dataframe().reset_index()
        ds.close()

        # Identify coordinate columns
        has_lat = "latitude" in df.columns
        has_lon = "longitude" in df.columns

        if not (has_lat and has_lon):
            raise ValueError(
                f"GRIB2 file missing latitude/longitude columns: {file_path}. "
                f"Available columns: {list(df.columns)}"
            )

        # Identify data columns (exclude metadata and non-meteorological)
        exclude_cols = {"latitude", "longitude", "time", "step", "valid_time",
                        "number", "surface", "heightAboveGround", "values"}
        data_cols = [c for c in df.columns if c not in exclude_cols]

        # Resolve valid_time: prefer explicit column, fall back to computation
        if "valid_time" in df.columns:
            df["forecast_time"] = pd.to_datetime(df["valid_time"])
        elif "step" in df.columns:
            df["forecast_time"] = basis_time + pd.to_timedelta(df["step"])
        else:
            raise ValueError(f"Cannot determine forecast_time from {file_path}")

        # Split by coordinate
        result: Dict[str, pd.DataFrame] = {}
        grouped = df.groupby(["latitude", "longitude"])

        for (lat, lon), group_df in grouped:
            coord_key = f"{lat}_{lon}"

            out_df = group_df[data_cols + ["forecast_time"]].copy()
            out_df = out_df.set_index("forecast_time").sort_index()
            out_df["basis_time"] = basis_time
            out_df["is_valid"] = True

            result[coord_key] = out_df

        return result

    @staticmethod
    def _extract_basis_time(ds: xr.Dataset, file_path: Path) -> pd.Timestamp:
        """Extract basis_time from dataset metadata, falling back to filename.

        Priority:
            1. ds.time coordinate (scalar or first value)
            2. Filename parsing (%Y-%m-%d_%H then %Y%m%d%H)
        """
        # Try dataset metadata first
        if "time" in ds.coords:
            time_val = ds.coords["time"].values
            if time_val.ndim == 0:
                return pd.Timestamp(time_val.item())
            elif len(time_val) > 0:
                return pd.Timestamp(time_val[0])

        # Fallback: parse from filename
        stem = file_path.stem
        for fmt in ["%Y-%m-%d_%H", "%Y%m%d%H"]:
            try:
                return pd.to_datetime(stem, format=fmt)
            except ValueError:
                continue

        raise ValueError(
            f"Cannot determine basis_time from dataset metadata or "
            f"filename '{file_path.name}'"
        )
