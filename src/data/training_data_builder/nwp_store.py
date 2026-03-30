"""
NWPDataStore — load-once cache for NWP data.

Reads all required NWP CSVs into a single ``(basis_time, forecast_time)``
MultiIndex DataFrame.  Subsequent queries for individual horizons or the
continuous format operate purely in memory.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures

import pandas as pd
from tqdm.auto import tqdm

from .config import NWPSourceConfig, ScadaConfig, TurbineInfo
from .resolvers.base import AbstractNWPResolver
from .time_alignment import (
    convert_timezone,
    create_nwp_basis_mapping,
    parse_frequency,
)

logger = logging.getLogger(__name__)


class NWPDataStore:
    """In-memory cache of NWP forecast data for a single turbine / farm.

    Internal representation
    -----------------------
    ``self._df`` is a DataFrame with a ``(basis_time, forecast_time)``
    MultiIndex (both in *NWP timezone*, typically UTC) and NWP variable
    columns.

    Public accessors transparently convert to the SCADA timezone before
    returning results.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        nwp_name: str,
        nwp_tz: str,
        scada_tz: str,
    ):
        self._df = df
        self.nwp_name = nwp_name
        self.nwp_tz = nwp_tz
        self.scada_tz = scada_tz

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        nwp_config: NWPSourceConfig,
        scada_config: ScadaConfig,
        resolver: AbstractNWPResolver,
        scada_kst_index: pd.DatetimeIndex,
        turbine_id: str,
    ) -> "NWPDataStore":
        """Load all NWP CSVs relevant to *scada_kst_index* for one turbine.

        Steps:
            1. Compute which NWP files (basis_times) are needed.
            2. Use *resolver* to locate and read each file.
            3. Optionally interpolate to match SCADA interval.
            4. Stack into a ``(basis_time, forecast_time)`` MultiIndex DF.
        """
        data_path = resolver.get_data_path(nwp_config.root, turbine_id)

        # Collect all needed basis_time strings
        # (We gather them from a full-range mapping so we don't miss any.)
        mapping = create_nwp_basis_mapping(
            scada_kst_index,
            nwp_config.frequency,
            scada_config.timezone,
            nwp_config.timezone,
        )

        # Only the basis_time keys are needed; the forecast-time lists are unused.
        basis_times = list(mapping.keys())

        dfs: List[pd.DataFrame] = []

        missing_basis_times: List[str] = []

        def _load_single(basis_time_str: str) -> Optional[pd.DataFrame]:
            try:
                df_single = resolver.load_basis_time(data_path, basis_time_str, turbine_id)
                basis_ts = pd.Timestamp(basis_time_str.replace("_", " "))
                df_single["basis_time"] = basis_ts
                return df_single
            except FileNotFoundError:
                missing_basis_times.append(basis_time_str)
                return None
            except Exception as e:
                logger.warning("Failed to load %s: %s", basis_time_str, e)
                return None

        # Determine thread count: use minimal CPU impact but high IO concurrency
        # Because we read tiny CSVs, threading dominates overhead.
        max_workers = min(32, (os.cpu_count() or 4) * 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_load_single, t_str): t_str
                for t_str in basis_times
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"[{nwp_config.name}] loading {turbine_id}",
                leave=False,
            ):
                res_df = future.result()
                if res_df is not None:
                    dfs.append(res_df)

        if missing_basis_times:
            missing_sorted = sorted(missing_basis_times)
            logger.warning(
                "[%s] turbine %s: %d/%d basis_times missing (first: %s, last: %s)",
                nwp_config.name,
                turbine_id,
                len(missing_sorted),
                len(basis_times),
                missing_sorted[0],
                missing_sorted[-1],
            )

        if not dfs:
            raise ValueError(
                f"No NWP data loaded for {nwp_config.name} at {data_path}"
            )

        combined = pd.concat(dfs)
        combined.index = pd.to_datetime(combined.index)
        combined.index.name = "forecast_time"
        combined = combined.reset_index().set_index(["basis_time", "forecast_time"])
        combined = combined.sort_index()

        # Fill missing basis_times from nearest available data
        combined = cls._fill_missing_basis_times(
            combined, missing_basis_times, nwp_config.name,
        )

        # Interpolate if forecast interval > scada interval
        combined = cls._interpolate_if_needed(
            combined, nwp_config.forecast_interval, scada_config.interval
        )

        # Prefix column names with NWP model name and enforce a single 'forecast_' prefix
        new_cols = {}
        for c in combined.columns:
            # Remove only the leading 'forecast_' if present
            base = c.removeprefix('forecast_')
            new_cols[c] = f"{nwp_config.name}_forecast_{base}"
        combined = combined.rename(columns=new_cols)

        return cls(
            df=combined,
            nwp_name=nwp_config.name,
            nwp_tz=nwp_config.timezone,
            scada_tz=scada_config.timezone,
        )

    @staticmethod
    def _fill_missing_basis_times(
        df: pd.DataFrame,
        missing_basis_times: List[str],
        nwp_name: str,
    ) -> pd.DataFrame:
        """Fill missing basis_times using the nearest available basis_time.

        For each missing basis_time, copies forecast data from the nearest
        previous (or next, if no previous exists) basis_time, shifts
        forecast_times by the time offset, and marks all rows as
        ``is_valid = False``.

        This ensures that ``get_for_horizon()`` can always find data for
        every SCADA timestamp, even when original NWP files were never
        downloaded.
        """
        if not missing_basis_times:
            return df

        available = df.index.get_level_values("basis_time").unique().sort_values()
        filled_groups = []
        unfillable = []

        for bt_str in sorted(missing_basis_times):
            bt = pd.Timestamp(bt_str.replace("_", " "))

            # Find nearest previous basis_time; fall back to nearest future
            earlier = available[available < bt]
            later = available[available > bt]
            if not earlier.empty:
                source_bt = earlier[-1]
            elif not later.empty:
                source_bt = later[0]
            else:
                unfillable.append(bt_str)
                continue

            source_data = df.loc[source_bt].copy()
            time_offset = bt - source_bt

            # Shift forecast_times to match the new basis_time's range
            new_forecast_times = source_data.index + time_offset
            source_data.index = new_forecast_times
            source_data.index.name = "forecast_time"

            if "is_valid" in source_data.columns:
                source_data["is_valid"] = False

            source_data["basis_time"] = bt
            source_data = source_data.reset_index().set_index(
                ["basis_time", "forecast_time"]
            )
            filled_groups.append(source_data)

        if filled_groups:
            total_rows = sum(len(g) for g in filled_groups)
            filled_cols = [
                c for c in filled_groups[0].columns if c != "is_valid"
            ]
            logger.warning(
                "[%s] Filled %d missing basis_times (%d rows, %d columns: %s) "
                "from nearest available data → is_valid=False",
                nwp_name,
                len(filled_groups),
                total_rows,
                len(filled_cols),
                ", ".join(filled_cols),
            )
            df = pd.concat([df] + filled_groups).sort_index()

        if unfillable:
            logger.error(
                "[%s] %d basis_times could not be filled (no data at all): %s",
                nwp_name,
                len(unfillable),
                ", ".join(unfillable),
            )

        return df

    @staticmethod
    def _interpolate_if_needed(
        df: pd.DataFrame,
        forecast_interval: str,
        scada_interval: str,
    ) -> pd.DataFrame:
        """Resample forecast rows within each basis_time group if needed.

        For example: 3h forecast data → 1h via linear interpolation.
        ``is_valid`` is forward-filled (not interpolated).
        """
        fc_hours, _ = parse_frequency(forecast_interval)
        sc_hours, _ = parse_frequency(scada_interval)

        if fc_hours <= sc_hours:
            return df  # Already fine or finer resolution

        is_valid_col = None
        if "is_valid" in df.columns:
            is_valid_col = "is_valid"

        groups = []
        for basis_time, group in df.groupby(level="basis_time"):
            # Drop to single-level index for resampling
            g = group.droplevel("basis_time")

            if is_valid_col:
                is_valid = g[is_valid_col].copy()
                g = g.drop(columns=[is_valid_col])

            # Resample to target interval and interpolate
            g = g.resample(scada_interval).interpolate(method="linear")

            if is_valid_col:
                is_valid = is_valid.astype("boolean").reindex(g.index).ffill().astype(bool)
                g[is_valid_col] = is_valid

            g["basis_time"] = basis_time
            g = g.reset_index().set_index(["basis_time", "forecast_time"])
            groups.append(g)

        return pd.concat(groups).sort_index()
        

    def get_for_horizon(
        self,
        horizon: int,
        scada_kst_index: pd.DatetimeIndex,
        nwp_frequency: str,
    ) -> pd.DataFrame:
        """Return NWP data for a specific forecast horizon.

        Returns a DataFrame indexed by SCADA time (KST) containing
        the NWP forecast variables for ``scada_time + horizon``, mapped
        to the most recent valid basis_time.
        """
        from .time_alignment import convert_timezone, snap_to_nwp_basis
        df = self._df.reset_index()

        scada_utc = convert_timezone(scada_kst_index, self.scada_tz, self.nwp_tz)
        avoid_exact = (horizon == 0)
        basis_utc = snap_to_nwp_basis(scada_utc, nwp_frequency, avoid_exact=avoid_exact)
        forecast_utc = scada_utc + pd.Timedelta(hours=horizon)

        targets = pd.DataFrame({
            "basis_time": basis_utc,
            "forecast_time": forecast_utc,
            "scada_kst": scada_kst_index,
        })

        merged = pd.merge(targets, df, on=["basis_time", "forecast_time"], how="left")
        merged = merged.set_index("scada_kst").drop(columns=["basis_time", "forecast_time"])
        # pd.merge may coerce the datetime dtype of the index — restore the original
        merged.index = scada_kst_index
        merged = merged.sort_index()

        # Fill missing NWP rows with same-column value from 24h before.
        nan_mask = merged.isna().any(axis=1)
        n_missing = int(nan_mask.sum())
        if n_missing > 0:
            data_cols = [c for c in merged.columns if c != "is_valid"]
            # Try -24h, -48h, -72h, ... expanding until filled
            max_days = len(merged) // 24 + 1
            for day_offset in range(1, max_days + 1):
                still_nan = merged[data_cols].isna().any(axis=1)
                if not still_nan.any():
                    break
                lookup_idx = merged.index[still_nan] - pd.Timedelta(hours=24 * day_offset)
                fill_vals = merged[data_cols].reindex(lookup_idx)
                fill_vals.index = merged.index[still_nan]
                merged.loc[still_nan, data_cols] = fill_vals

            # If still NaN after exhausting all previous days, error
            final_nan = merged[data_cols].isna().any(axis=1)
            if final_nan.any():
                nan_cols = merged[data_cols].columns[
                    merged.loc[final_nan, data_cols].isna().any()
                ].tolist()
                nan_times = merged.index[final_nan].tolist()
                raise ValueError(
                    f"{self.nwp_name}: horizon={horizon}, "
                    f"{int(final_nan.sum())} rows could not be filled from any previous day. "
                    f"Columns: {nan_cols}, "
                    f"Times: {nan_times[:5]}{'...' if len(nan_times) > 5 else ''}. "
                    f"More NWP data is needed before these dates."
                )

            if "is_valid" in merged.columns:
                merged.loc[nan_mask, "is_valid"] = False

            logger.warning(
                "%s: horizon=%d, %d/%d rows filled from previous day (is_valid=False)",
                self.nwp_name, horizon, n_missing, len(merged),
            )

        return merged

    def get_for_continuous(self) -> pd.DataFrame:
        """Return NWP data for the continuous (single-file) format.

        For each SCADA timestamp, selects the forecast from the most
        recent NWP release.  The output is indexed by time in SCADA
        timezone (KST).
        """
        df = self._df.reset_index()

        # Convert forecast_time to SCADA tz for the output
        df["time"] = convert_timezone(
            pd.DatetimeIndex(df["forecast_time"]),
            self.nwp_tz,
            self.scada_tz,
        )

        # For each (time), keep the row from the most recent basis_time
        # (i.e. largest basis_time ≤ corresponding scada-utc time)
        df = df.sort_values(["time", "basis_time"])
        df = df.drop_duplicates(subset="time", keep="last")

        df = df.set_index("time").drop(columns=["basis_time", "forecast_time"])
        df = df.sort_index()
        return df


def merge_nwp_stores(
    stores: List[NWPDataStore],
    nwp_configs: List["NWPSourceConfig"],
    mode: str,
    horizon: int = 0,
    scada_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Merge data from multiple NWP stores.

    Args:
        stores: List of NWPDataStore instances (different NWP sources).
        nwp_configs: List of NWPSourceConfigs corresponding to the stores.
        mode: ``"per_horizon"`` or ``"continuous"``.
        horizon: Forecast horizon (only used in ``"per_horizon"`` mode).
        scada_index: The SCADA timestamps to align to (needed for per_horizon).

    Returns:
        Merged DataFrame with columns prefixed by NWP name.
    """
    if not stores:
        raise ValueError("No NWP stores provided")
    if len(stores) != len(nwp_configs):
        raise ValueError(
            f"stores ({len(stores)}) and nwp_configs ({len(nwp_configs)}) "
            f"must have the same length"
        )

    dfs = []
    for store, config in zip(stores, nwp_configs):
        if mode == "per_horizon":
            if scada_index is None:
                raise ValueError("scada_index is required for per_horizon mode")
            dfs.append(store.get_for_horizon(horizon, scada_index, config.frequency))

        elif mode == "continuous":
            dfs.append(store.get_for_continuous())
        else:
            raise ValueError(f"Unknown mode: {mode}")

    merged = pd.concat(dfs, axis=1)

    # Safety net: if NaN remains after per-source filling (should be rare),
    # fill with same-column value from 24h before, then ffill+bfill.
    nan_mask = merged.isna().any(axis=1)
    n_nan_rows = int(nan_mask.sum())
    if n_nan_rows > 0:
        nan_cols = merged.columns[merged.isna().any()].tolist()
        logger.warning(
            "merge_nwp_stores: %d rows with NaN after concat "
            "(horizon=%d, columns=%s). Filling from previous day.",
            n_nan_rows, horizon, nan_cols,
        )
        data_cols = [c for c in merged.columns if "is_valid" not in c]
        max_days = len(merged) // 24 + 1
        for day_offset in range(1, max_days + 1):
            still_nan = merged[data_cols].isna().any(axis=1)
            if not still_nan.any():
                break
            shifted = merged[data_cols].shift(24 * day_offset, freq="h")
            merged.loc[still_nan, data_cols] = shifted.loc[still_nan, data_cols]

        final_nan = merged[data_cols].isna().any(axis=1)
        if final_nan.any():
            unfilled_cols = merged[data_cols].columns[
                merged.loc[final_nan, data_cols].isna().any()
            ].tolist()
            nan_times = merged.index[final_nan].tolist()
            raise ValueError(
                f"merge_nwp_stores: horizon={horizon}, "
                f"{int(final_nan.sum())} rows could not be filled from any previous day. "
                f"Columns: {unfilled_cols}, "
                f"Times: {nan_times[:5]}{'...' if len(nan_times) > 5 else ''}. "
                f"More NWP data is needed before these dates."
            )

        is_valid_cols = [c for c in merged.columns if "is_valid" in c]
        if is_valid_cols:
            merged.loc[nan_mask, is_valid_cols] = False

    return merged
