"""
Abstract base class for NWP data resolvers.

A resolver knows *which* subdirectory or data subset to use for a given
turbine.  Concrete implementations handle coordinate-based selection
(ECMWF) and pressure-level / hub-height selection (KMA).
"""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class AbstractNWPResolver(ABC):
    """Strategy interface for selecting NWP data per turbine."""

    @abstractmethod
    def get_data_path(self, nwp_root: Path, turbine_id: str) -> Path:
        """Return the directory containing CSVs for *turbine_id*.

        For coordinate-based resolvers this is the nearest ``{lat}_{lon}``
        directory; for pressure-level resolvers this is *nwp_root* itself
        (all levels are needed).
        """

    @abstractmethod
    def load_basis_time(
        self,
        data_path: Path,
        basis_time_str: str,
        turbine_id: str,
    ) -> pd.DataFrame:
        """Load a single basis-time file and return a DataFrame indexed
        by ``forecast_time`` with the columns appropriate for *turbine_id*.
        """
