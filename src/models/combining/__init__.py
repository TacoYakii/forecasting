"""Forecast combining module.

Provides combiners that merge multiple models' ForecastResult objects
into a single combined forecast using various strategies.
"""

from .base import BaseCombiner
from .equal_weight import EqualWeightCombiner
from .horizontal import HorizontalCombiner
from .vertical import VerticalCombiner

__all__ = [
    "BaseCombiner",
    "EqualWeightCombiner",
    "HorizontalCombiner",
    "VerticalCombiner",
]
