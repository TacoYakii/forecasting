"""
Training data pipeline orchestrator.

Coordinates:
    1. Resolver creation (per NWP source).
    2. NWPDataStore loading (per source × per turbine).
    3. SCADA loading.
    4. Builder invocation (per_horizon / continuous).
    5. Farm-level aggregation from turbine-level data.
"""
import logging
from datetime import datetime
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .builders.continuous import ContinuousBuilder
from .builders.per_horizon import PerHorizonBuilder
from .config import TrainingDataConfig
from .nwp_store import NWPDataStore, merge_nwp_stores
from .resolvers.base import AbstractNWPResolver
from .resolvers.coordinate import CoordinateResolver
from .resolvers.pressure_level import PressureLevelResolver
from .scada import load_scada

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Farm-level aggregation
# ---------------------------------------------------------------------------


def aggregate_to_farm(
    turbine_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Average turbine-level DataFrames into a farm-level DataFrame.

    Numeric columns are averaged; ``is_valid`` is the AND across all
    turbines.
    """
    if not turbine_dfs:
        raise ValueError("No turbine DataFrames to aggregate")

    # Get a common index (intersection of all indices)
    common_index: pd.Index | None = None
    for df in turbine_dfs.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    if common_index is None or common_index.empty:
        raise ValueError("No common time index found across turbine DataFrames.")

    # Reindex all DataFrames and validate
    is_valid_parts = []
    numeric_parts = []

    for tid, df in turbine_dfs.items():
        aligned = df.reindex(common_index)

        nan_per_col = aligned.isna().sum()
        n_missing = int(nan_per_col.sum())
        if n_missing > 0:
            bad_cols = nan_per_col[nan_per_col > 0]
            logger.error(
                "Turbine '%s': %d NaN after index alignment. Per-column: %s",
                tid, n_missing, bad_cols.to_dict(),
            )
            raise ValueError(
                f"Turbine '{tid}' has {n_missing} missing values after index alignment. "
                f"Columns with NaN: {bad_cols.to_dict()}"
            )

        if "is_valid" in aligned.columns:
            is_valid_parts.append(aligned["is_valid"])
            numeric_parts.append(aligned.drop(columns=["is_valid"]))
        else:
            numeric_parts.append(aligned)

    # Average numeric columns — use numpy mean to preserve the index dtype exactly
    if not numeric_parts:
        raise ValueError("No numeric data to aggregate after processing 'is_valid' columns.")

    cols = numeric_parts[0].columns
    stacked = np.stack([df.values for df in numeric_parts], axis=0)  # (n_turbines, n_rows, n_cols)
    combined_numeric = pd.DataFrame(
        stacked.mean(axis=0),
        index=common_index,
        columns=cols,
    )

    # Combine 'is_valid' columns with logical AND
    if is_valid_parts:
        combined_is_valid = pd.concat(is_valid_parts, axis=1).all(axis=1)
        combined_is_valid.index = common_index
        combined_numeric["is_valid"] = combined_is_valid

    return combined_numeric


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TrainingDataPipeline:
    """High-level orchestrator for building training datasets.

    Farm-level data is always derived by averaging turbine-level NWP data.

    Usage::

        config = TrainingDataConfig(...)
        pipeline = TrainingDataPipeline(config)
        pipeline.build(output_format="per_horizon", level="farm")
        pipeline.build(output_format="continuous", level="turbine")
    """

    def __init__(
        self,
        config: TrainingDataConfig,
        distance_metric: Literal["euclidean", "haversine"] = "euclidean",
    ):
        self.config = config
        self.distance_metric = distance_metric
        self._resolvers: Dict[str, AbstractNWPResolver] = {}
        self._setup_file_logging()
        self._create_resolvers()

    def _setup_file_logging(self) -> None:
        """Configure file logging for the training_data_builder package.

        Writes to ``{output_dir}/build_{timestamp}.log`` with real-time
        flushing so logs are visible immediately during long runs.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.config.output_dir / f"build_{timestamp}.log"

        # Attach handler to the package root logger (works regardless of import path)
        pkg_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
        pkg_logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

        # Flush every line immediately
        handler.stream.reconfigure(line_buffering=True)

        pkg_logger.addHandler(handler)
        self._log_handler = handler
        logger.info("Log file: %s", log_path)

    def _create_resolvers(self) -> None:
        """Instantiate a resolver for each NWP source."""
        for src in self.config.nwp_sources:
            if src.resolver_type == "coordinate":
                self._resolvers[src.name] = CoordinateResolver(
                    self.config.turbine_info,
                    distance_metric=self.distance_metric,
                )
            elif src.resolver_type == "pressure_level":
                self._resolvers[src.name] = PressureLevelResolver(
                    self.config.turbine_info,
                )
            else:
                raise ValueError(
                    f"Unknown resolver_type '{src.resolver_type}' "
                    f"for NWP source '{src.name}'"
                )

    # ------------------------------------------------------------------
    # NWP loading helpers
    # ------------------------------------------------------------------

    def _load_nwp_stores(
        self,
        turbine_id: str,
        scada_index: pd.DatetimeIndex,
    ) -> List[NWPDataStore]:
        """Load NWP data stores for a single turbine."""
        stores = []
        for src in self.config.nwp_sources:
            store = NWPDataStore.load(
                nwp_config=src,
                scada_config=self.config.scada,
                resolver=self._resolvers[src.name],
                scada_kst_index=scada_index,
                turbine_id=turbine_id,
            )
            stores.append(store)
        return stores

    # ------------------------------------------------------------------
    # Turbine-level building
    # ------------------------------------------------------------------

    def _build_turbine(
        self,
        turbine_id: str,
        output_format: str,
    ) -> None:
        """Build data for a single turbine."""
        scada_path = self.config.scada.turbine_level_path(turbine_id)
        scada = load_scada(scada_path)

        nwp_stores = self._load_nwp_stores(turbine_id, scada.index)

        if output_format == "per_horizon":
            out_dir = self.config.output_dir / "per_horizon" / "turbine_level" / f"turbine_{turbine_id}"
            PerHorizonBuilder().build(scada, self.config, out_dir, nwp_stores=nwp_stores)

        elif output_format == "continuous":
            out_path = self.config.output_dir / "continuous" / "turbine_level" / f"turbine_{turbine_id}.csv"
            ContinuousBuilder().build(scada, self.config, out_path, nwp_stores=nwp_stores)

    # ------------------------------------------------------------------
    # Farm-level building (= turbine NWP averaged)
    # ------------------------------------------------------------------

    def _load_farm_turbine_nwp(
        self,
        farm_scada_index: pd.DatetimeIndex,
    ) -> Dict[str, List[NWPDataStore]]:
        """Load NWP stores for all turbines (once, shared by farm builders)."""
        turbine_ids = list(self.config.turbine_info.keys())
        turbine_nwp: Dict[str, List[NWPDataStore]] = {}
        for tid in tqdm(turbine_ids, desc="Loading NWP for turbines"):
            turbine_nwp[tid] = self._load_nwp_stores(tid, farm_scada_index)
        return turbine_nwp

    def _make_farm_nwp_provider(
        self,
        turbine_nwp: Dict[str, List[NWPDataStore]],
        scada_index: pd.DatetimeIndex,
    ):
        """Create an NWP provider that aggregates turbine NWP into farm-level.

        Returns a callable ``(horizon: int) → pd.DataFrame``.
        """
        turbine_ids = list(turbine_nwp.keys())
        nwp_configs = self.config.nwp_sources

        def provider(horizon: int) -> pd.DataFrame:
            turbine_dfs = {}
            for tid in turbine_ids:
                turbine_dfs[tid] = merge_nwp_stores(
                    turbine_nwp[tid],
                    nwp_configs,
                    "per_horizon",
                    horizon=horizon,
                    scada_index=scada_index,
                )
            return aggregate_to_farm(turbine_dfs)

        return provider

    def _build_farm(self, output_format: str) -> None:
        """Build farm-level data by averaging turbine NWP, delegating to builders."""
        farm_scada = load_scada(self.config.scada.farm_level_path())
        scada_index = pd.DatetimeIndex(farm_scada.index)
        turbine_nwp = self._load_farm_turbine_nwp(scada_index)
        farm_nwp_provider = self._make_farm_nwp_provider(turbine_nwp, scada_index)

        if output_format == "per_horizon":
            out_dir = self.config.output_dir / "per_horizon" / "farm_level"
            PerHorizonBuilder().build(
                farm_scada, self.config, out_dir,
                nwp_provider=farm_nwp_provider,
            )

        elif output_format == "continuous":
            out_path = self.config.output_dir / "continuous" / "farm_level.csv"
            # Continuous uses horizon=0 internally
            ContinuousBuilder().build(
                farm_scada, self.config, out_path,
                nwp_provider=lambda: farm_nwp_provider(0),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        output_format: Literal["per_horizon", "continuous"] = "per_horizon",
        level: Literal["farm", "turbine"] = "farm",
    ) -> None:
        """Build training datasets.

        Args:
            output_format: ``"per_horizon"`` or ``"continuous"``.
            level: ``"farm"`` (turbine NWP averaged) or ``"turbine"``.
        """
        logger.info("Building %s / %s level", output_format, level)

        if level == "turbine":
            turbine_ids = list(self.config.turbine_info.keys())
            if not turbine_ids:
                raise ValueError(
                    "turbine_info is empty. Turbine-level builds require "
                    "turbine metadata (lat, lon, hub_height) for NWP resolution."
                )
            for tid in tqdm(turbine_ids, desc="Turbines"):
                self._build_turbine(tid, output_format)

        elif level == "farm":
            self._build_farm(output_format)

        else:
            raise ValueError(f"Unknown level: {level}")

        logger.info("Complete: %s / %s", output_format, level)

    def build_all(self) -> None:
        """Build all combinations of format × level."""
        for fmt in ("per_horizon", "continuous"):
            for lvl in ("farm", "turbine"):
                self.build(output_format=fmt, level=lvl)
