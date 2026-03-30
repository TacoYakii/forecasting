"""
Abstract base class for NWP data readers.

All readers convert provider-specific raw formats into a common
Dict[str, pd.DataFrame] interface so downstream processors are
format-agnostic.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd


class AbstractReader(ABC):
    """
    Abstract base for NWP data readers.

    Each concrete reader handles a specific file format (GRIB2, KMA TXT, etc.)
    and converts raw files into a common dictionary of DataFrames.

    The output contract:
        - Keys: coordinate identifier strings (e.g., "33.3_126.2" or point ID)
        - Values: pd.DataFrame with:
            - index: forecast_time (DatetimeIndex)
            - columns: meteorological variable columns + "basis_time"
    """

    @abstractmethod
    def read(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Read a single raw NWP file and return per-coordinate DataFrames.

        Args:
            file_path: Path to the raw NWP file.

        Returns:
            Dictionary mapping coordinate keys to their DataFrames.
            - GRIB2: multiple grid points → multiple entries
            - KMA TXT: single point → single entry
        """

    @abstractmethod
    def validate_file(self, file_path: Path) -> bool:
        """
        Check file integrity before reading.

        Args:
            file_path: Path to the file to validate.

        Returns:
            True if file is valid and readable, False otherwise.
        """

    @abstractmethod
    def list_files(self, input_dir: Path) -> List[Path]:
        """
        Return sorted list of files to process in the given directory.

        Each reader knows its own file extension(s) and handles
        discovery internally.

        Args:
            input_dir: Directory to search for raw NWP files.

        Returns:
            Sorted list of file paths to process.
        """
