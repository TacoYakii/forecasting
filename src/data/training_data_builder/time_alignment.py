"""
Time alignment utilities for matching SCADA timestamps to NWP files.

All public functions accept and return KST-based timestamps unless
explicitly noted otherwise.  UTC conversions are handled internally.
"""
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from .config import get_timezone_offset_hours


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_frequency(frequency: str) -> Tuple[int, str]:
    """Parse a frequency string such as ``"6h"`` → ``(6, "h")``.

    Only hour-based units are supported.
    """
    match = re.match(r"(\d+)([a-zA-Z]+)", frequency)
    if not match:
        raise ValueError(f"Invalid frequency format: {frequency}")

    value = int(match.group(1))
    raw_unit = match.group(2).lower()

    hour_units = {"h", "hr", "hour", "hours"}
    if raw_unit not in hour_units:
        raise ValueError(
            f"Only hour units are supported.  Got '{raw_unit}'.  "
            f"Supported: {hour_units}"
        )
    return value, "h"


# ---------------------------------------------------------------------------
# Timezone conversion
# ---------------------------------------------------------------------------


def convert_timezone(
    index: pd.DatetimeIndex,
    source_tz: str,
    target_tz: str,
) -> pd.DatetimeIndex:
    """Shift a *naive* DatetimeIndex from *source_tz* to *target_tz*.

    Example::

        kst_index = convert_timezone(utc_index, "UTC", "KST")  # +9h
    """
    offset_hours = get_timezone_offset_hours(source_tz, target_tz)
    return index + pd.Timedelta(hours=offset_hours)


# ---------------------------------------------------------------------------
# NWP basis-time snapping
# ---------------------------------------------------------------------------


def snap_to_nwp_basis(
    utc_times: pd.DatetimeIndex,
    frequency: str,
    *,
    avoid_exact: bool = False,
) -> pd.DatetimeIndex:
    """Snap UTC timestamps to the nearest *previous* NWP release time.

    Args:
        utc_times: Timestamps in UTC.
        frequency: NWP release cycle (e.g. ``"6h"``).
        avoid_exact: If ``True``, when a timestamp exactly aligns with an NWP
            release, snap to the *previous* release instead (prevents
            observation leakage for horizon-0).

    Returns:
        DatetimeIndex of NWP basis times (UTC).
    """
    # Determine the hour offset based on the NWP release frequency.
    freq_hours, _ = parse_frequency(frequency)
    remainder = utc_times.hour % freq_hours

    if avoid_exact:
        # When the remainder is 0 the time sits exactly on a release
        # boundary → push it back one full cycle.
        adjusted = remainder.where(remainder != 0, freq_hours)
        snapped = utc_times - pd.to_timedelta(adjusted, unit="h")
    else:
        snapped = utc_times - pd.to_timedelta(remainder, unit="h")

    # Ensure minutes, seconds, and microseconds are zeroed out.
    return snapped.floor("h")


# ---------------------------------------------------------------------------
# Index mapping: SCADA time (KST) → NWP file name + forecast times
# ---------------------------------------------------------------------------


def create_nwp_file_mapping(
    scada_kst_index: pd.DatetimeIndex,
    nwp_frequency: str,
    scada_tz: str,
    nwp_tz: str,
    forecasting_horizon: int = 0,
) -> Dict[str, List[str]]:
    """Build a mapping from NWP file stems to the forecast-time rows
    that should be extracted from each file.

    The mapping accounts for:
    1. KST → UTC conversion (using ``scada_tz`` / ``nwp_tz``).
    2. Snapping to the nearest NWP release cycle.
    3. Computing forecast times = SCADA-UTC + horizon.

    Args:
        scada_kst_index: SCADA DatetimeIndex in ``scada_tz``.
        nwp_frequency: NWP release cycle (e.g. ``"6h"``).
        scada_tz: Timezone of the SCADA data (e.g. ``"KST"``).
        nwp_tz: Timezone of the NWP data (e.g. ``"UTC"``).
        forecasting_horizon: Forecast horizon in hours.

    Returns:
        ``{file_stem: [forecast_time_str, ...]}``
        where *file_stem* has format ``"YYYY-MM-DD_HH"`` (NWP timezone)
        and *forecast_time_str* has format ``"YYYY-MM-DD HH:00:00"``
        (also in NWP timezone, matching CSV index).
    """
    # Convert SCADA time → NWP timezone
    scada_in_nwp_tz = convert_timezone(scada_kst_index, scada_tz, nwp_tz)

    # Snap to NWP basis times
    avoid_exact = forecasting_horizon == 0
    nwp_basis_times = snap_to_nwp_basis(
        scada_in_nwp_tz, nwp_frequency, avoid_exact=avoid_exact
    )

    # Forecast time = scada UTC time + horizon
    forecast_times = scada_in_nwp_tz + pd.Timedelta(hours=forecasting_horizon)

    # Build mapping
    basis_strs = nwp_basis_times.strftime("%Y-%m-%d_%H")
    forecast_strs = forecast_times.strftime("%Y-%m-%d %H:00:00")

    mapping: Dict[str, List[str]] = defaultdict(list)
    for basis_str, fc_str in zip(basis_strs, forecast_strs):
        mapping[basis_str].append(fc_str)

    return dict(mapping)


def create_nwp_basis_mapping(
    scada_kst_index: pd.DatetimeIndex,
    nwp_frequency: str,
    scada_tz: str,
    nwp_tz: str,
) -> Dict[str, List[str]]:
    """Build a mapping from NWP file stems to the forecast-time rows
    needed for the **continuous** (single-file) output format.

    For each SCADA timestamp, uses the nearest *previous* NWP release
    (with ``avoid_exact=True`` to prevent observation leakage) and maps
    the SCADA timestamp to the corresponding NWP forecast time.

    Returns:
        ``{file_stem: [forecast_time_str, ...]}``
        Both in NWP timezone.
    """
    scada_in_nwp_tz = convert_timezone(scada_kst_index, scada_tz, nwp_tz)

    nwp_basis_times = snap_to_nwp_basis(
        scada_in_nwp_tz, nwp_frequency, avoid_exact=True
    )

    basis_strs = nwp_basis_times.strftime("%Y-%m-%d_%H")
    forecast_strs = scada_in_nwp_tz.strftime("%Y-%m-%d %H:00:00")

    mapping: Dict[str, List[str]] = defaultdict(list)
    for basis_str, fc_str in zip(basis_strs, forecast_strs):
        mapping[basis_str].append(fc_str)

    return dict(mapping)
