"""
Configuration dataclasses for the training data builder pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------

TIMEZONE_OFFSETS: Dict[str, int] = {
    "UTC": 0,
    "KST": 9,
}


def get_timezone_offset_hours(source_tz: str, target_tz: str) -> int:
    """Return the offset in hours to convert *source_tz* → *target_tz*.

    Example:
        get_timezone_offset_hours("UTC", "KST") → 9
        get_timezone_offset_hours("KST", "UTC") → -9
    """
    if source_tz not in TIMEZONE_OFFSETS:
        raise ValueError(
            f"Unknown timezone '{source_tz}'. Supported: {list(TIMEZONE_OFFSETS)}"
        )
    if target_tz not in TIMEZONE_OFFSETS:
        raise ValueError(
            f"Unknown timezone '{target_tz}'. Supported: {list(TIMEZONE_OFFSETS)}"
        )
    return TIMEZONE_OFFSETS[target_tz] - TIMEZONE_OFFSETS[source_tz]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScadaConfig:
    """SCADA data source configuration.

    Attributes:
        root: Directory containing SCADA CSVs.
              Expected structure:
                root/
                ├── farm_level.csv
                └── turbine_level/
                    └── turbine_{id}.csv
        interval: SCADA observation interval (e.g. ``"1h"``).
        timezone: Timezone of SCADA timestamps (e.g. ``"KST"``).
    """

    root: Path
    interval: str = "1h"
    timezone: str = "KST"

    def __post_init__(self):
        self.root = Path(self.root)
        if self.timezone not in TIMEZONE_OFFSETS:
            raise ValueError(
                f"Unsupported timezone '{self.timezone}'. "
                f"Supported: {list(TIMEZONE_OFFSETS)}"
            )

    def farm_level_path(self) -> Path:
        return self.root / "farm_level.csv"

    def turbine_level_path(self, turbine_id: str) -> Path:
        return self.root / "turbine_level" / f"turbine_{turbine_id}.csv"

    def list_turbine_ids(self) -> List[str]:
        """Discover turbine IDs from filenames in turbine_level/."""
        turbine_dir = self.root / "turbine_level"
        if not turbine_dir.exists():
            raise FileNotFoundError(f"Turbine directory not found: {turbine_dir}")
        return sorted(
            f.stem.replace("turbine_", "")
            for f in turbine_dir.glob("turbine_*.csv")
        )


@dataclass
class NWPSourceConfig:
    """Configuration for a single NWP data source.

    Attributes:
        name: Human-readable identifier (e.g. ``"ECMWF"``, ``"GDAPS"``).
        root: Directory containing preprocessed NWP CSVs.
            For ``resolver_type="coordinate"``:
                root/{lat}_{lon}/{basis_time}.csv
            For ``resolver_type="pressure_level"``:
                root/{group}/{pressure_level}/{basis_time}.csv
                where {group} is the KMA data-point group (A, B, C, ...)
                assigned to each turbine via ``TurbineInfo.nwp_group``.
        frequency: NWP release cycle (e.g. ``"6h"`` = 4 times per day).
        forecast_interval: Time step of forecast rows within a single CSV
            (e.g. ``"1h"`` or ``"3h"``).
        timezone: Timezone of the NWP forecast timestamps (typically ``"UTC"``).
        resolver_type: How to select data for individual turbines.
            ``"coordinate"``      → nearest lat/lon directory.
            ``"pressure_level"``  → nearest hub-height pressure level.
    """

    name: str
    root: Path
    frequency: str = "6h"
    forecast_interval: str = "1h"
    timezone: str = "UTC"
    resolver_type: Literal["coordinate", "pressure_level"] = "coordinate"

    def __post_init__(self):
        self.root = Path(self.root)
        if self.timezone not in TIMEZONE_OFFSETS:
            raise ValueError(
                f"Unsupported timezone '{self.timezone}'. "
                f"Supported: {list(TIMEZONE_OFFSETS)}"
            )
        valid_resolvers = {"coordinate", "pressure_level"}
        if self.resolver_type not in valid_resolvers:
            raise ValueError(
                f"Invalid resolver_type '{self.resolver_type}'. "
                f"Must be one of {valid_resolvers}"
            )


@dataclass
class TurbineInfo:
    """Physical metadata for a single wind turbine.

    Attributes:
        lat: Latitude (decimal degrees).
        lon: Longitude (decimal degrees).
        hub_height: Hub height above ground (metres).
        nwp_group: KMA data-point group identifier (e.g. ``"A"``, ``"B"``).
            Used by ``PressureLevelResolver`` to select the correct
            subdirectory.  ``None`` for non-KMA sources.
    """

    lat: float
    lon: float
    hub_height: float
    nwp_group: str | None = None


@dataclass
class TrainingDataConfig:
    """Top-level configuration for the training data pipeline.

    Attributes:
        scada: SCADA data source configuration.
        nwp_sources: One or more NWP data sources to merge.
        output_dir: Directory for generated training data.
        turbine_info: Physical metadata for each turbine.  NWP data is
            resolved per-turbine; farm level is derived by averaging.
        max_lag: Maximum number of lag hours for SCADA features.
        max_forecast_horizon: Maximum forecast horizon (hours).
        positive_columns: Columns to clip at zero (e.g. power output).
    """

    scada: ScadaConfig
    nwp_sources: List[NWPSourceConfig]
    output_dir: Path
    turbine_info: Dict[str, TurbineInfo] = field(default_factory=dict)
    max_lag: int = 6
    max_forecast_horizon: int = 49
    positive_columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class TemporalHierarchyConfig:
    """Configuration for temporal hierarchy dataset generation.

    Temporal hierarchy aggregates base per-horizon CSVs into lower-frequency
    block averages. For frequency *k*, consecutive groups of *k* base horizons
    are averaged to produce ``max_forecast_horizon / k`` output horizons.

    Attributes:
        frequencies: Aggregation frequencies. Each must evenly divide
            ``max_forecast_horizon`` (e.g. ``[1, 2, 4, 6, 12, 24, 48]``
            for a 48-horizon base).
        y_col: Target column for observed values.
        index_col: Column used to align DataFrames across horizons.
        target_index_col: Column representing the forecast target timestamp.
        is_valid_col: Boolean validity flag column.

    Example:
        >>> config = TemporalHierarchyConfig(frequencies=[1, 2, 4, 8, 12, 24, 48])
    """

    frequencies: List[int]
    y_col: str = "forecast_time_observed_KPX_pwr"
    index_col: str = "basis_time"
    target_index_col: str = "forecast_time"
    is_valid_col: Optional[str] = "is_valid"

