"""Base configuration with YAML serialization.

All model configs inherit from BaseConfig to get type-safe fields
and standardized save/load via YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class with YAML save/load.

    Subclasses define typed fields via ``@dataclass``.  Serialization
    converts all values to plain Python types (tuples → lists, Paths → str)
    so the resulting YAML is portable and human-readable.

    Example:
        >>> @dataclass
        ... class MyConfig(BaseConfig):
        ...     learning_rate: float = 0.001
        ...     hidden_size: int = 128
        >>> cfg = MyConfig(learning_rate=0.01)
        >>> cfg.save(Path("config.yaml"))
        >>> loaded = MyConfig.load(Path("config.yaml"))
    """

    def save(self, path: Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Destination file path (.yaml).
        """
        data = self._to_serializable(asdict(self))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """Load configuration from a YAML file.

        Args:
            path: Source file path (.yaml).

        Returns:
            A new config instance with values from the file.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Restore tuples for fields annotated as Tuple
        data = cls._restore_types(data)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary.

        Returns:
            Dict with all fields as serializable Python types.
        """
        return self._to_serializable(asdict(self))

    @classmethod
    def _restore_types(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Restore tuple fields that were saved as lists."""
        type_hints = {f.name: f.type for f in fields(cls)}
        restored = {}
        for key, value in data.items():
            if key in type_hints and isinstance(value, list):
                hint = type_hints[key]
                if isinstance(hint, str) and "Tuple" in hint:
                    value = tuple(value)
                elif hasattr(hint, "__origin__") and hint.__origin__ is tuple:
                    value = tuple(value)
            restored[key] = value
        return restored

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Recursively convert to YAML-safe types."""
        if isinstance(obj, dict):
            return {k: BaseConfig._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BaseConfig._to_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj
