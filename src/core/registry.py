"""Model registry for mapping string names to forecaster classes.

All forecaster classes register themselves via the ``@MODEL_REGISTRY.register_model``
decorator so that they can be looked up by name at runtime.

Example:
    >>> from src.core.registry import MODEL_REGISTRY
    >>> model_cls = MODEL_REGISTRY.get("xgboost")
    >>> model_cls
    <class 'src.models.machine_learning.xgboost_model.XGBoostForecaster'>
"""

from typing import Dict, Optional, Type


class Registry:
    """Name-to-class mapping for forecaster models.

    Example:
        >>> registry = Registry("Models")
        >>> @registry.register_model(name="my_model")
        ... class MyForecaster: ...
        >>> registry.get("my_model")
        <class 'MyForecaster'>
    """

    def __init__(self, name: str):
        """Initialize a registry with the given name.

        Args:
            name: Human-readable name for error messages.

        Example:
            >>> registry = Registry("Models")
        """
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def register_model(self, name: Optional[str] = None):
        """Decorator that registers a class under the given name.

        Args:
            name: Registry key. Defaults to ``cls.__name__.lower()``.

        Example:
            >>> @MODEL_REGISTRY.register_model(name="xgboost")
            ... class XGBoostForecaster: ...
        """
        def _register(cls):
            model_name = name if name is not None else cls.__name__.lower()
            if model_name in self._module_dict:
                raise KeyError(
                    f"{model_name} is already registered in {self._name}"
                )
            self._module_dict[model_name] = cls
            cls._registry_key = model_name
            return cls
        return _register

    def get(self, name: str) -> Type:
        """Retrieve a registered class by name.

        Args:
            name: Registry key.

        Returns:
            The registered class.

        Raises:
            KeyError: If the name is not registered.

        Example:
            >>> MODEL_REGISTRY.get("ngboost")
            <class '...NGBoostForecaster'>
        """
        if name not in self._module_dict:
            raise KeyError(
                f"{name} is not registered. "
                f"Available models: {list(self._module_dict.keys())}"
            )
        return self._module_dict[name]


MODEL_REGISTRY = Registry("Models")
