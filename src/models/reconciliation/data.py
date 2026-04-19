"""Data preparation utilities for hierarchical reconciliation.

Reconciliation uses fully materialized numpy arrays rather than
DataLoaders.  All strategies consume the same input structure:
``ReconciliationData``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

import numpy as np


@dataclass
class ReconciliationData:
    """Standard input container for reconciliation.

    Attributes:
        keys: Sorted time keys used to align forecast and observed data.
        forecast: Forecast array with shape ``(T, N, Q)`` for
            probabilistic reconciliation or ``(T, N)`` for deterministic
            reconciliation.
        observed: Observed array with shape ``(T, N)``.
    """

    keys: list[Hashable]
    forecast: np.ndarray
    observed: np.ndarray


def _to_2d_observed(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"observed array must be 1-D or 2-D, got shape {arr.shape}")


def prepare_reconciliation_data(
    forecast_dict: dict,
    observed_dict: dict,
    is_valid_dict: dict | None = None,
) -> ReconciliationData:
    """Align dictionaries on common sorted keys and return numpy arrays."""
    common_keys = set(forecast_dict) & set(observed_dict)
    if is_valid_dict is not None:
        common_keys &= set(is_valid_dict)

    if not common_keys:
        raise ValueError("No common timestamps found across forecast/observed data.")

    sorted_keys = sorted(common_keys)
    valid_keys: list[Hashable] = []
    for key in sorted_keys:
        if is_valid_dict is None or np.all(np.asarray(is_valid_dict[key])):
            valid_keys.append(key)

    if not valid_keys:
        raise ValueError("No valid timestamps remain after is_valid filtering.")

    forecast = np.stack([np.asarray(forecast_dict[k], dtype=float) for k in valid_keys], axis=0)
    observed = np.stack([_to_2d_observed(np.asarray(observed_dict[k])) for k in valid_keys], axis=0)

    if observed.ndim != 3 or observed.shape[-1] != 1:
        raise ValueError(
            "Observed arrays must align to shape (T, N, 1) before squeezing, "
            f"got {observed.shape}."
        )
    observed = observed[..., 0]

    if forecast.shape[:2] != observed.shape:
        raise ValueError(
            "Forecast and observed arrays do not align on (T, N): "
            f"forecast {forecast.shape}, observed {observed.shape}."
        )

    return ReconciliationData(keys=valid_keys, forecast=forecast, observed=observed)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_reconciliation_splits(
    model_input_root: Path,
    observed_root: Path,
) -> tuple[ReconciliationData, ReconciliationData, ReconciliationData]:
    """Load ``base_train_data``, ``recon_val_data`` and ``test_data``.

    Expected files under ``model_input_root``:
    - ``base_train/sampled_forecasts.pkl``
    - ``validation/sampled_forecasts.pkl``
    - ``test/sampled_forecasts.pkl``

    Expected files under ``observed_root``:
    - ``observed_values.pkl``
    - ``is_valid.pkl``
    """
    model_input_root = Path(model_input_root)
    observed_root = Path(observed_root)

    observed_dict = _load_pickle(observed_root / "observed_values.pkl")
    is_valid_dict = _load_pickle(observed_root / "is_valid.pkl")

    split_to_dir = {
        "base_train": model_input_root / "base_train" / "sampled_forecasts.pkl",
        "recon_val": model_input_root / "validation" / "sampled_forecasts.pkl",
        "test": model_input_root / "test" / "sampled_forecasts.pkl",
    }

    loaded = {
        name: prepare_reconciliation_data(
            _load_pickle(path), observed_dict, is_valid_dict
        )
        for name, path in split_to_dir.items()
    }
    return loaded["base_train"], loaded["recon_val"], loaded["test"]
