"""
Continuous (single-file) training data builder.

Generates one CSV per entity (farm or turbine), containing:
    - time (index, KST)
    - SCADA observation columns
    - NWP forecast variables from the most recent NWP release
    - is_valid flag
"""
import logging
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from ..config import TrainingDataConfig
from ..nwp_store import NWPDataStore, merge_nwp_stores
from ..scada import clip_positive_columns
from .utils import format_final_dataset

logger = logging.getLogger(__name__)

# Callable that returns NWP DataFrame (no arguments — always horizon=0)
NWPContinuousProvider = Callable[[], pd.DataFrame]


class ContinuousBuilder:
    """Build a single continuous time-series CSV.

    Output:
        One CSV file with ``time`` as the index (KST), SCADA columns,
        and the best-available NWP forecast at each timestamp.
    """

    def build(
        self,
        scada: pd.DataFrame,
        config: TrainingDataConfig,
        output_path: Path,
        *,
        nwp_stores: Optional[List[NWPDataStore]] = None,
        nwp_provider: Optional[NWPContinuousProvider] = None,
    ) -> None:
        """Generate the continuous CSV file.

        Provide exactly one of *nwp_stores* or *nwp_provider*.

        Args:
            scada: SCADA DataFrame (KST index).
            config: Pipeline configuration.
            output_path: Full path for the output CSV.
            nwp_stores: Loaded NWP data stores (turbine-level default).
            nwp_provider: Custom callable ``() → NWP DataFrame``
                (used for farm-level aggregation).
        """
        if (nwp_stores is None) == (nwp_provider is None):
            raise ValueError("Provide exactly one of nwp_stores or nwp_provider")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use exactly horizon=0 logic (which prevents data leakage!)
        if nwp_provider is not None:
            nwp_merged = nwp_provider()
        else:
            nwp_merged = merge_nwp_stores(
                nwp_stores,
                config.nwp_sources,
                "per_horizon",
                horizon=0,
                scada_index=scada.index,
            )

        # For continuous format, SCADA columns are observations at the current time
        # (not forecasting targets), so use original column names without prefix.
        scada_obs = scada.drop(columns=["is_valid"], errors="ignore")
        scada_is_valid = scada["is_valid"] if "is_valid" in scada.columns else None

        # Merge SCADA + NWP on time index
        merged = pd.concat(
            [scada_obs, nwp_merged],
            axis=1,
            join="inner",
        )

        # Combine is_valid from SCADA and NWP
        if scada_is_valid is not None:
            merged["scada_is_valid"] = scada_is_valid.reindex(merged.index)

        if config.positive_columns:
            merged = clip_positive_columns(merged, config.positive_columns)

        merged.index.name = "basis_time"
        merged = format_final_dataset(merged, config.nwp_sources, is_continuous=True)

        # Save
        merged.to_csv(output_path)
        logger.info("Written continuous file: %s (%d rows)", output_path, len(merged))
