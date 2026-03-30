"""
Per-horizon training data builder.

Generates one CSV per forecast horizon, each containing:
    - basis_time (index, KST)
    - forecast_time
    - NWP forecast variables (prefixed by model name)
    - SCADA lagged features
    - SCADA target at forecast_time
    - is_valid flag
"""
import logging
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from ..config import TrainingDataConfig
from ..nwp_store import NWPDataStore, merge_nwp_stores
from ..scada import (
    clip_positive_columns,
    create_lagged_features,
    prepare_target,
)
from .utils import format_final_dataset

logger = logging.getLogger(__name__)

# Callable that takes a horizon (int) and returns NWP DataFrame
NWPPerHorizonProvider = Callable[[int], pd.DataFrame]


class PerHorizonBuilder:
    """Build per-horizon CSV files.

    Output structure::

        {output_dir}/
        ├── horizon_0.csv
        ├── horizon_1.csv
        └── ...
    """

    def build(
        self,
        scada: pd.DataFrame,
        config: TrainingDataConfig,
        output_dir: Path,
        *,
        nwp_stores: Optional[List[NWPDataStore]] = None,
        nwp_provider: Optional[NWPPerHorizonProvider] = None,
    ) -> None:
        """Generate per-horizon CSV files.

        Provide exactly one of *nwp_stores* or *nwp_provider*.

        Args:
            scada: SCADA DataFrame (KST index).
            config: Pipeline configuration.
            output_dir: Directory to write ``horizon_N.csv`` files.
            nwp_stores: Loaded NWP data stores (turbine-level default).
            nwp_provider: Custom callable ``(horizon) → NWP DataFrame``
                (used for farm-level aggregation).
        """
        if (nwp_stores is None) == (nwp_provider is None):
            raise ValueError("Provide exactly one of nwp_stores or nwp_provider")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute lagged features once
        lagged = create_lagged_features(scada, config.max_lag)

        def _get_nwp(horizon: int) -> pd.DataFrame:
            if nwp_provider is not None:
                return nwp_provider(horizon)
            return merge_nwp_stores(
                nwp_stores,
                config.nwp_sources,
                "per_horizon",
                horizon=horizon,
                scada_index=scada.index,
            )

        for horizon in tqdm(
            range(config.max_forecast_horizon),
            desc=f"  Horizons → {output_dir.name}",
        ):
            nwp_merged = _get_nwp(horizon)

            # Target at forecast_time
            target = prepare_target(scada, horizon)

            # Merge all components on shared index
            merged = pd.concat(
                [lagged, nwp_merged, target],
                axis=1,
                join="inner",
            )

            merged["forecast_time"] = merged.index + pd.Timedelta(hours=horizon)
            merged.index.name = "basis_time"

            if config.positive_columns:
                merged = clip_positive_columns(merged, config.positive_columns)

            merged = format_final_dataset(merged, config.nwp_sources, is_continuous=False)

            # Save
            out_path = output_dir / f"horizon_{horizon}.csv"
            merged.to_csv(out_path)

        logger.info("Written %d horizon files to %s", config.max_forecast_horizon, output_dir)
