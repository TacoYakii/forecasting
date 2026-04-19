"""Adapter for converting ForecastResult objects into NMAPEEvaluator inputs.

NMAPEEvaluator historically consumed per-horizon CSV files with
``basis_time``/``forecast_time``/``mu`` columns. This module bridges the
in-memory ForecastResult architecture to that schema by extracting the
distribution mean as the point forecast for each horizon.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.core.forecast_results import BaseForecastResult


def to_nmape_frames(
    result: BaseForecastResult,
    step: pd.Timedelta = pd.Timedelta(hours=1),
    horizons: Optional[List[int]] = None,
    point: str = "mean",
) -> Dict[int, pd.DataFrame]:
    """Convert a ForecastResult into per-horizon DataFrames for NMAPEEvaluator.

    For each requested horizon ``h`` the returned DataFrame has columns
    ``basis_time``, ``forecast_time``, ``mu`` where ``mu`` is the chosen
    point summary of the forecast distribution at horizon ``h`` and
    ``forecast_time = basis_time + h * step``.

    Args:
        result: Any BaseForecastResult subclass
            (Parametric/Quantile/Sample/Grid). The point forecast is taken
            from ``result.to_distribution(h)`` for every horizon.
        step: Time interval represented by one horizon step. The default
            ``pd.Timedelta(hours=1)`` matches the KPX nMAPE rules hard-coded
            into NMAPEEvaluator (hourly basis filters and horizon ranges).
        horizons: Subset of 1-indexed horizons to convert. ``None`` converts
            all horizons from 1 to ``result.horizon``.
        point: Which point summary of the distribution to use as ``mu``.
            ``"mean"`` uses ``distribution.mean()`` (default, back-compat);
            ``"median"`` uses ``distribution.ppf(0.5)``.

    Returns:
        Mapping from horizon (int) to a DataFrame with columns
        ``basis_time``, ``forecast_time``, ``mu``.

    Raises:
        TypeError: If ``result.basis_index`` is not a DatetimeIndex.
        ValueError: If any horizon in ``horizons`` is outside [1, H] or
            ``point`` is not one of ``{"mean", "median"}``.

    Example:
        >>> frames = to_nmape_frames(result)
        >>> frames[1].columns.tolist()
        ['basis_time', 'forecast_time', 'mu']
        >>> frames_rt = to_nmape_frames(result, horizons=[2])
        >>> frames_med = to_nmape_frames(result, point="median")
    """
    if point not in ("mean", "median"):
        raise ValueError(
            f"point must be 'mean' or 'median', got {point!r}."
        )
    if not isinstance(result.basis_index, pd.DatetimeIndex):
        raise TypeError(
            "to_nmape_frames requires result.basis_index to be a "
            f"pd.DatetimeIndex, got {type(result.basis_index).__name__}."
        )

    H = result.horizon
    if horizons is None:
        horizons = list(range(1, H + 1))
    else:
        for h in horizons:
            if h < 1 or h > H:
                raise ValueError(
                    f"horizon {h} out of range [1, {H}] for this result."
                )

    basis = result.basis_index
    frames: Dict[int, pd.DataFrame] = {}
    for h in horizons:
        dist = result.to_distribution(h)
        mu = dist.mean() if point == "mean" else dist.ppf(0.5)
        frames[h] = pd.DataFrame(
            {
                "basis_time": basis,
                "forecast_time": basis + h * step,
                "mu": mu,
            }
        )
    return frames
