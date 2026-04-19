"""nMAPE evaluation for wind power forecasting (KPX rules).

This package exposes:
    NMAPEEvaluator: KPX Day-Ahead / Real-Time nMAPE evaluator.
    JejuNMAPEEvaluator: Jeju-specific nMAPE evaluator.
    to_nmape_frames: Adapter from ForecastResult to per-horizon DataFrames
        consumable by NMAPEEvaluator via ``forecast_frames``.
"""

from .adapter import to_nmape_frames
from .evaluator import NMAPEEvaluator
from .jeju import JejuNMAPEEvaluator

__all__ = ["NMAPEEvaluator", "JejuNMAPEEvaluator", "to_nmape_frames"]
