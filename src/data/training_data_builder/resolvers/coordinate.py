"""
Coordinate-based NWP resolver for ECMWF-style data.

Selects the nearest lat/lon grid-point directory for each turbine.
"""
import logging
import math
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd

from ..config import TurbineInfo
from .base import AbstractNWPResolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def _euclidean(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _haversine(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0  # Earth radius in km
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


_DISTANCE_FUNCTIONS = {
    "euclidean": _euclidean,
    "haversine": _haversine,
}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class CoordinateResolver(AbstractNWPResolver):
    """Select the nearest ECMWF coordinate directory for each turbine.

    The NWP root is expected to contain subdirectories named
    ``{lat}_{lon}`` (e.g. ``"35.0_126.5"``).

    Caches parsed coordinate directories on first access to avoid
    re-scanning and re-parsing the NWP root on every call.

    Args:
        turbine_info: Mapping of turbine IDs to their metadata.
        distance_metric: ``"euclidean"`` (default) or ``"haversine"``.
        drop_columns: Columns to drop from loaded CSVs (e.g. ``["values"]``).
    """

    def __init__(
        self,
        turbine_info: Dict[str, TurbineInfo],
        distance_metric: Literal["euclidean", "haversine"] = "euclidean",
        drop_columns: list[str] | None = None,
    ):
        if distance_metric not in _DISTANCE_FUNCTIONS:
            raise ValueError(
                f"Unknown distance_metric '{distance_metric}'.  "
                f"Supported: {list(_DISTANCE_FUNCTIONS)}"
            )
        self.turbine_info = turbine_info
        self._dist_fn = _DISTANCE_FUNCTIONS[distance_metric]
        self.drop_columns = drop_columns or ["values"]

        # Cache: parsed coordinate directories for a given nwp_root
        self._nwp_root_cached: Path | None = None
        self._coord_dirs: List[Tuple[float, float, str]] = []

    # -- internal helpers ------------------------------------------------

    def _ensure_coord_cache(self, nwp_root: Path) -> None:
        """Scan and parse coordinate directories once per nwp_root."""
        if self._nwp_root_cached == nwp_root:
            return

        coords: List[Tuple[float, float, str]] = []
        for d in nwp_root.iterdir():
            if not d.is_dir():
                continue
            try:
                parts = d.name.split("_")
                lat, lon = float(parts[0]), float(parts[1])
                coords.append((lat, lon, d.name))
            except (ValueError, IndexError):
                continue

        if not coords:
            raise FileNotFoundError(
                f"No coordinate directories found in {nwp_root}"
            )

        self._coord_dirs = coords
        self._nwp_root_cached = nwp_root
        logger.debug("Cached %d coordinate directories from %s", len(coords), nwp_root)

    # -- interface -------------------------------------------------------

    def get_data_path(self, nwp_root: Path, turbine_id: str) -> Path:
        """Return the base NWP root directory."""
        return nwp_root

    def load_basis_time(
        self,
        data_path: Path,
        basis_time_str: str,
        turbine_id: str,
    ) -> pd.DataFrame:
        """Find the nearest coordinate directory that contains the
        requested basis time and load it.

        Uses the cached coordinate list to avoid re-scanning the
        directory tree on every call.
        """
        self._ensure_coord_cache(data_path)
        info = self.turbine_info[turbine_id]

        # Filter to coordinate dirs that have the requested file
        available = [
            (lat, lon, name)
            for lat, lon, name in self._coord_dirs
            if (data_path / name / f"{basis_time_str}.csv").exists()
        ]

        if not available:
            raise FileNotFoundError(
                f"No coordinate directories contain data for {basis_time_str} "
                f"in {data_path}"
            )

        # Find nearest among those that actually have the file
        best = min(
            available,
            key=lambda c: self._dist_fn((info.lat, info.lon), (c[0], c[1])),
        )
        best_dir = best[2]
        best_dist = self._dist_fn((info.lat, info.lon), (best[0], best[1]))

        # Warn if the nearest available coordinate is far from the turbine
        # (suggests the true nearest coordinate is missing this basis_time)
        nearest_overall = min(
            self._coord_dirs,
            key=lambda c: self._dist_fn((info.lat, info.lon), (c[0], c[1])),
        )
        nearest_dist = self._dist_fn((info.lat, info.lon), (nearest_overall[0], nearest_overall[1]))
        if best_dir != nearest_overall[2]:
            logger.warning(
                "Turbine %s @ %s: nearest coord %s (dist=%.4f) missing file, "
                "fell back to %s (dist=%.4f)",
                turbine_id, basis_time_str,
                nearest_overall[2], nearest_dist,
                best_dir, best_dist,
            )

        csv_path = data_path / best_dir / f"{basis_time_str}.csv"
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "forecast_time"

        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df
