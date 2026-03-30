from typing import Dict, Type, Optional


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def register_model(self, name: Optional[str] = None):
        def _register(cls):
            model_name = name if name is not None else cls.__name__.lower()
            if model_name in self._module_dict:
                raise KeyError(f"{model_name} is already registered in {self._name}")
            self._module_dict[model_name] = cls
            return cls
        return _register

    def get(self, name: str) -> Type:
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered. Available models: {list(self._module_dict.keys())}")
        return self._module_dict[name]


MODEL_REGISTRY = Registry("Models")
