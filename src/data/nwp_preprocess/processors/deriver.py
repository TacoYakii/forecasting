"""
Derived variable computation for NWP data.

Consolidates wind speed/direction calculations from the legacy KMA derive.py
and ECMWF derive() into a single, reusable processor.

Supports multiple height-level wind components (e.g., u10/v10, u100/v100),
producing derived variables named by height (wspd10, wdir10, wspd100, wdir100).
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VariableDeriver:
    """
    Computes derived meteorological variables from raw NWP outputs.

    Supports multiple height-level wind components simultaneously.

    Supported derived variables:
        - "wspd": wind speed from U and V components (per height level)
        - "wdir": wind direction (meteorological convention) from U and V (per height level)
        - "height": geometric height from geopotential height

    Args:
        wind_components: Dict mapping height suffix to (u_col, v_col) pairs.
            e.g., {"10": ("u10", "v10"), "100": ("u100", "v100")}
        geopotential_col: Column name for geopotential height.
    """

    GRAVITATIONAL_ACCELERATION = 9.80665

    def __init__(
        self,
        wind_components: Optional[Dict[str, Tuple[str, str]]] = None,
        geopotential_col: str = "geo_potential_height",
    ):
        self.wind_components = wind_components or {"10": ("u10", "v10")}
        self.geopotential_col = geopotential_col

    @staticmethod
    def _compute_wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
        """Compute wind speed: sqrt(u² + v²)."""
        return np.sqrt(u ** 2 + v ** 2)

    @staticmethod
    def _compute_wind_direction(u: pd.Series, v: pd.Series) -> pd.Series:
        """Compute wind direction in meteorological convention (degrees).

        Direction the wind is coming FROM, measured clockwise from north.
        Returns values in [0, 360).
        """
        return (180 + np.degrees(np.arctan2(u, v))) % 360

    def compute_geometric_height(self, df: pd.DataFrame) -> pd.Series:
        """Convert geopotential height to geometric height.

        Uses metpy if available, otherwise falls back to simple approximation.
        """
        try:
            import metpy.calc
            from metpy.units import units

            geopotential = (
                df[self.geopotential_col].values * self.GRAVITATIONAL_ACCELERATION
            )
            geopotential_qty = units.Quantity(geopotential, "m**2/s**2")
            height = metpy.calc.geopotential_to_height(geopotential_qty)
            return pd.Series(height.magnitude, index=df.index, name="height")

        except ImportError:
            logger.warning(
                "metpy not installed. Using simple geopotential/g approximation."
            )
            return df[self.geopotential_col]

    def apply_all(
        self,
        df: pd.DataFrame,
        derive_variables: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply all requested derived variable computations.

        For wind variables (wspd, wdir), iterates over all height levels
        in wind_components and creates columns like wspd10, wdir10,
        wspd100, wdir100, etc.

        Args:
            df: Input DataFrame with raw NWP variables.
            derive_variables: List of variables to derive.
                Supported: "wspd", "wdir", "height".
                If None, defaults to ["wspd", "wdir"].

        Returns:
            DataFrame with derived columns added.
        """
        if derive_variables is None:
            derive_variables = ["wspd", "wdir"]

        df = df.copy()

        # Process wind components for each height level
        for height_suffix, (u_col, v_col) in self.wind_components.items():
            has_u = u_col in df.columns
            has_v = v_col in df.columns

            if not (has_u and has_v):
                if "wspd" in derive_variables or "wdir" in derive_variables:
                    logger.warning(
                        "Cannot compute wind variables for height '%s': "
                        "columns '%s' and/or '%s' not found.",
                        height_suffix, u_col, v_col,
                    )
                continue

            if "wspd" in derive_variables:
                df[f"wspd{height_suffix}"] = self._compute_wind_speed(
                    df[u_col], df[v_col]
                )

            if "wdir" in derive_variables:
                df[f"wdir{height_suffix}"] = self._compute_wind_direction(
                    df[u_col], df[v_col]
                )

        # Height derivation
        if "height" in derive_variables:
            if self.geopotential_col in df.columns:
                df["height"] = self.compute_geometric_height(df)
            else:
                logger.warning(
                    "Cannot compute height: column '%s' not found.",
                    self.geopotential_col,
                )

        return df
