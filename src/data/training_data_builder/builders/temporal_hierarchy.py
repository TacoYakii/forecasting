"""Temporal hierarchy training data builder.

Generates block-averaged per-horizon CSVs at multiple temporal frequencies
from a set of base (1-hourly) per-horizon CSV files.

For frequency *k*, consecutive groups of *k* base horizons are averaged::

    freq=2:  horizon_1 = avg(base_h1, base_h2)
             horizon_2 = avg(base_h3, base_h4)
             ...

Output structure::

    {output_dir}/
    ├── 1/
    │   ├── horizon_1.csv   (copy of base)
    │   └── ...
    ├── 2/
    │   ├── horizon_1.csv
    │   └── ...
    └── 48/
        └── horizon_1.csv
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import TemporalHierarchyConfig
from .utils import recompute_wind_derived

logger = logging.getLogger(__name__)


def _validate_frequencies(
    frequencies: List[int], max_horizon: int
) -> None:
    """Check that every frequency evenly divides *max_horizon*.

    Args:
        frequencies: Requested aggregation frequencies.
        max_horizon: Number of base horizon files (e.g. 48).

    Raises:
        ValueError: If any frequency does not divide *max_horizon* or is < 1.
    """
    for freq in frequencies:
        if freq < 1:
            raise ValueError(
                f"Frequency must be a positive integer, got {freq}"
            )
        if max_horizon % freq != 0:
            raise ValueError(
                f"Frequency {freq} does not evenly divide "
                f"max_horizon {max_horizon}. "
                f"Valid divisors: {_divisors(max_horizon)}"
            )


def _divisors(n: int) -> List[int]:
    """Return sorted list of positive divisors of *n*."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _average_horizons(
    dfs: List[pd.DataFrame],
    config: TemporalHierarchyConfig,
) -> pd.DataFrame:
    """Compute block average of *k* horizon DataFrames.

    Averaging rules:
        - ``index_col`` (basis_time): preserved as common index.
        - ``target_index_col`` (forecast_time): max across the *k* frames.
        - ``is_valid_col``: logical AND across the *k* frames.
        - All other numeric columns: arithmetic mean.

    Args:
        dfs: List of *k* DataFrames to average.
        config: Column naming configuration.

    Returns:
        Averaged DataFrame with the same column structure.

    Example:
        >>> avg = _average_horizons([df_h1, df_h2], config)
    """
    if len(dfs) == 1:
        return dfs[0].copy()

    index_col = config.index_col
    target_index_col = config.target_index_col
    is_valid_col = config.is_valid_col

    # Align by index_col
    aligned = [
        df.set_index(index_col) if index_col in df.columns else df
        for df in dfs
    ]

    common_idx = aligned[0].index
    for df in aligned[1:]:
        common_idx = common_idx.intersection(df.index)

    if common_idx.empty:
        raise ValueError(f"No common {index_col} found across DataFrames")

    aligned = [df.loc[common_idx] for df in aligned]

    # forecast_time: take max (string max works for ISO format)
    target_indices = pd.concat(
        [df[target_index_col] for df in aligned], axis=1
    )
    max_target = target_indices.max(axis=1)

    # is_valid: AND
    has_valid = bool(is_valid_col and is_valid_col in aligned[0].columns)
    if has_valid:
        valid_combined = pd.concat(
            [df[is_valid_col] for df in aligned], axis=1
        ).all(axis=1)

    # Average numeric columns
    drop_cols = [target_index_col]
    if has_valid:
        drop_cols.append(is_valid_col)

    cols_to_avg = [c for c in aligned[0].columns if c not in drop_cols]
    stacked = np.stack([df[cols_to_avg].values for df in aligned])
    avg_data = np.mean(stacked, axis=0)

    result = pd.DataFrame(avg_data, columns=cols_to_avg, index=common_idx)
    result[target_index_col] = max_target
    if has_valid:
        result[is_valid_col] = valid_combined

    # Recompute wspd/wdir from averaged U/V components
    result = recompute_wind_derived(result)

    result = result.reset_index(names=[index_col])

    # Reorder: index_col, target_index_col, averaged cols, is_valid
    final_cols = [index_col, target_index_col] + cols_to_avg
    if has_valid:
        final_cols.append(is_valid_col)

    return result[final_cols]


def _block_average_continuous(
    df: pd.DataFrame,
    freq: int,
    config: TemporalHierarchyConfig,
) -> pd.DataFrame:
    """Block-average a continuous DataFrame by groups of *freq* consecutive rows.

    Averaging rules (consistent with ``_average_horizons``):
        - ``index_col`` (basis_time): last timestamp in each block.
        - ``is_valid_col``: logical AND across the block.
        - All other numeric columns: arithmetic mean.
        - Wind direction: recomputed from averaged U/V via
          ``recompute_wind_derived``.

    Rows that do not fill a complete block at the tail are dropped.

    Args:
        df: Continuous DataFrame with ``index_col`` as a column,
            sorted by time in ascending order.
        freq: Block size (number of consecutive rows to average).
        config: Column naming configuration.

    Returns:
        Block-averaged DataFrame.

    Example:
        >>> avg = _block_average_continuous(df, freq=4, config=config)
    """
    if freq == 1:
        return df.copy()

    index_col = config.index_col
    is_valid_col = config.is_valid_col

    n_complete = (len(df) // freq) * freq
    trimmed = df.iloc[:n_complete].copy()

    group_ids = np.arange(n_complete) // freq

    # Separate special columns
    result_index = trimmed.groupby(group_ids)[index_col].last()

    has_valid = bool(is_valid_col and is_valid_col in trimmed.columns)
    if has_valid:
        result_valid = trimmed.groupby(group_ids)[is_valid_col].all()

    # Average numeric columns
    drop_cols = [index_col]
    if has_valid:
        drop_cols.append(is_valid_col)

    num_cols = [c for c in trimmed.columns if c not in drop_cols]
    result = trimmed[num_cols].groupby(group_ids).mean()

    result[index_col] = result_index.values
    if has_valid:
        result[is_valid_col] = result_valid.values

    result = recompute_wind_derived(result)

    # Reorder: index_col first, then numeric, then is_valid
    final_cols = [index_col] + num_cols
    if has_valid:
        final_cols.append(is_valid_col)
    result = result[final_cols]

    return result.reset_index(drop=True)


class TemporalHierarchyBuilder:
    """Build temporal hierarchy CSVs via block average of base per-horizon data.

    For each frequency *k* in the config, reads consecutive groups of *k*
    base horizon CSVs from ``source_dir`` and writes averaged results to
    ``{output_dir}/{k}/horizon_{h}.csv``.

    Example:
        >>> builder = TemporalHierarchyBuilder()
        >>> builder.build(
        ...     source_dir=Path("data/standard/farm_level"),
        ...     output_dir=Path("data/hierarchy/temporal"),
        ...     max_horizon=48,
        ...     config=TemporalHierarchyConfig(frequencies=[1, 2, 4, 12, 48]),
        ... )
    """

    def build(
        self,
        source_dir: Path,
        output_dir: Path,
        max_horizon: int,
        config: TemporalHierarchyConfig,
    ) -> None:
        """Generate temporal hierarchy CSV files.

        Base horizons are always 1-indexed (``horizon_1.csv`` through
        ``horizon_{max_horizon}.csv``).  ``horizon_0.csv``, if present,
        is ignored because it represents the current time step rather
        than a forecast target.

        Args:
            source_dir: Directory containing base ``horizon_{h}.csv`` files.
            output_dir: Root output directory. Creates ``{freq}/`` subdirs.
            max_horizon: Number of base horizons to aggregate over.
            config: Temporal hierarchy configuration.

        Raises:
            ValueError: If any frequency does not divide *max_horizon*.
            FileNotFoundError: If a required base horizon CSV is missing.
        """
        _validate_frequencies(config.frequencies, max_horizon)

        # Pre-load base horizon DataFrames (1-indexed, horizon_0 excluded)
        logger.info("Loading %d base horizon files from %s", max_horizon, source_dir)
        base_dfs = {}
        for h in range(1, max_horizon + 1):
            path = source_dir / f"horizon_{h}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Base horizon file not found: {path}")
            base_dfs[h] = pd.read_csv(path)

        for freq in tqdm(config.frequencies, desc="Temporal hierarchy"):
            n_output = max_horizon // freq
            freq_dir = output_dir / str(freq)
            freq_dir.mkdir(parents=True, exist_ok=True)

            for out_h in range(1, n_output + 1):
                # Gather the k base horizons for this output horizon
                start_h = (out_h - 1) * freq + 1
                group = [base_dfs[start_h + i] for i in range(freq)]

                averaged = _average_horizons(group, config)
                out_path = freq_dir / f"horizon_{out_h}.csv"
                averaged.to_csv(out_path, index=False)

            logger.info(
                "freq=%d: written %d horizon files to %s",
                freq, n_output, freq_dir,
            )

    def build_continuous(
        self,
        source_path: Path,
        output_dir: Path,
        config: TemporalHierarchyConfig,
    ) -> None:
        """Generate temporal hierarchy continuous CSV files.

        Reads the base continuous CSV (1-hour resolution) and produces
        one block-averaged CSV per frequency under ``{output_dir}/{freq}.csv``.

        Args:
            source_path: Path to base continuous CSV
                (e.g. ``continuous/farm_level.csv``).
            output_dir: Root output directory for hierarchy continuous files.
            config: Temporal hierarchy configuration.

        Raises:
            FileNotFoundError: If *source_path* does not exist.

        Example:
            >>> builder = TemporalHierarchyBuilder()
            >>> builder.build_continuous(
            ...     source_path=Path("data/dongbok/continuous/farm_level.csv"),
            ...     output_dir=Path("data/dongbok/hierarchy/continuous"),
            ...     config=TemporalHierarchyConfig(frequencies=[1, 2, 4, 12, 48]),
            ... )
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Base continuous file not found: {source_path}")

        logger.info("Loading base continuous file: %s", source_path)
        base_df = pd.read_csv(source_path)

        output_dir.mkdir(parents=True, exist_ok=True)

        for freq in tqdm(config.frequencies, desc="Temporal hierarchy (continuous)"):
            averaged = _block_average_continuous(base_df, freq, config)
            out_path = output_dir / f"{freq}.csv"
            averaged.to_csv(out_path, index=False)
            logger.info(
                "freq=%d: written %d rows to %s",
                freq, len(averaged), out_path,
            )
