"""Persistence functions for loading saved models and forecast results.

This module provides factory functions that reconstruct model instances
from saved artifacts (runner_config.yaml + serialized model files).

Functions:
    load_model: Restore a fitted model from a save directory.
    load_forecast_result: Re-exported from forecast_results for convenience.

Example:
    >>> from src.core.persistence import load_model, load_forecast_result
    >>> model = load_model("res/arima_garch/exp_0/")
    >>> result = load_forecast_result("res/arima_garch/exp_0/forecast_result/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import yaml

from .forecast_results import load_forecast_result  # noqa: F401 — re-export


def load_model(
    save_dir: Union[str, Path],
    state: str = "fitted",
) -> object:
    """Restore a fitted model from a save directory.

    Reads runner_config.yaml to determine the model class via registry_key,
    creates an instance with the saved hyperparameters and model_name,
    then loads the serialized model weights via _load_model_specific().

    Args:
        save_dir: Directory containing runner_config.yaml and model/ subdirectory.
        state: Which model state to load.
            "fitted" -- fit() state before rolling (default).
            "post_rolling" -- state after rolling evaluation.
            Ignored for PerHorizonRunner (horizon models have a single state).

    Returns:
        For RollingRunner: a single BaseForecaster instance.
        For PerHorizonRunner: a Dict[int, BaseForecaster] mapping horizon to model.

    Raises:
        FileNotFoundError: If runner_config.yaml is missing.
        KeyError: If the registry_key is not registered.

    Example:
        >>> model = load_model("res/arima_garch/exp_0/")
        >>> runner = RollingRunner(model=model, dataset=new_df, ...)

        >>> models = load_model("res/ngboost/exp_0/")
        >>> models[1].forecast(X, index)  # horizon-1 model
    """
    save_dir = Path(save_dir)
    config_path = save_dir / "runner_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    from .registry import MODEL_REGISTRY

    runner_type = config["runner_type"]
    registry_key = config["registry_key"]
    hp = config.get("hyperparameter", {})
    display_name = config.get("model_name")

    model_cls = MODEL_REGISTRY.get(registry_key)

    if runner_type == "PerHorizonRunner":
        return _load_per_horizon_models(
            save_dir, model_cls, registry_key, hp, display_name, config,
        )
    else:
        return _load_single_model(
            save_dir, model_cls, registry_key, hp, display_name, state,
        )


def _load_single_model(
    save_dir: Path,
    model_cls,
    registry_key: str,
    hp: dict,
    display_name: str | None,
    state: str,
):
    """Load a single model (RollingRunner pattern).

    Constructs the model with hyperparameter and model_name, then loads
    model-specific state from the appropriate state directory.

    Args:
        save_dir: Root save directory.
        model_cls: The model class from registry.
        registry_key: Registry key string.
        hp: Hyperparameters dict.
        display_name: User-specified display name (restored as model_name).
        state: "fitted" or "post_rolling".

    Returns:
        A fitted model instance.

    Example:
        >>> model = _load_single_model(save_dir, cls, "arima_garch", {}, None, "fitted")
    """
    model = _create_model_instance(model_cls, hp, display_name)
    model_stem = save_dir / "model" / state / f"{registry_key}_model"
    model.load_model(model_stem)
    return model


def _load_per_horizon_models(
    save_dir: Path,
    model_cls,
    registry_key: str,
    hp: dict,
    display_name: str | None,
    config: dict,
) -> Dict[int, object]:
    """Load all per-horizon models (PerHorizonRunner pattern).

    Iterates over horizon_* directories under save_dir/model/ and loads
    each horizon's model.

    Args:
        save_dir: Root save directory.
        model_cls: The model class from registry.
        registry_key: Registry key string.
        hp: Hyperparameters dict.
        display_name: User-specified display name.
        config: Full runner_config dict (for horizons list).

    Returns:
        Dict mapping horizon int to fitted model instance.

    Example:
        >>> models = _load_per_horizon_models(save_dir, cls, "ngboost", {}, None, cfg)
        >>> models[1].forecast(X, index)
    """
    models: Dict[int, object] = {}
    model_dir = save_dir / "model"

    # Use horizons from config if available, otherwise discover from directories
    if "horizons" in config and config["horizons"]:
        horizon_list = config["horizons"]
    else:
        import re
        horizon_list = []
        for d in model_dir.iterdir():
            m = re.match(r"^horizon_(\d+)$", d.name)
            if m and d.is_dir():
                horizon_list.append(int(m.group(1)))
        horizon_list.sort()

    for h in horizon_list:
        model = _create_model_instance(model_cls, hp, display_name)
        h_dir = model_dir / f"horizon_{h}"
        model_stem = h_dir / f"{registry_key}_model"
        model.load_model(model_stem)
        models[h] = model

    return models


def _create_model_instance(model_cls, hp: dict, display_name: str | None):
    """Create a model instance with the BaseForecaster interface.

    Args:
        model_cls: The model class to instantiate.
        hp: Hyperparameters dict.
        display_name: User-specified display name (model_name parameter).

    Returns:
        An uninitialized model instance.

    Example:
        >>> model = _create_model_instance(XGBoostForecaster, {"n_estimators": 100}, None)
    """
    kwargs = {"hyperparameter": hp}
    if display_name is not None:
        kwargs["model_name"] = display_name
    return model_cls(**kwargs)
