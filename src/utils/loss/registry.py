from typing import Dict, Type, Any, Optional
import torch.nn as nn


class LossRegistry:
    """Registry to keep track of available loss functions."""

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        """Decorator to register a loss function class."""

        def inner_wrapper(wrapped_class: Type[nn.Module]) -> Type[nn.Module]:
            registry_name = name if name is not None else wrapped_class.__name__
            cls._registry[registry_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str, **kwargs) -> nn.Module:
        """Retrieve a registered loss function by name."""
        if name not in cls._registry:
            raise ValueError(
                f"Loss '{name}' not found. Available losses: {list(cls._registry.keys())}"
            )
        loss_class = cls._registry[name]
        return loss_class(**kwargs)


def get_loss_from_config(config: Any, **kwargs) -> nn.Module:
    """
    Creates a loss function instance based on the configuration.

    Args:
        config: Configuration object containing 'loss_func' attribute (e.g., TrainerConfig).
        **kwargs: Additional arguments to pass to the loss function constructor.
                    (e.g., 'quantiles' for QuantileCRPSLoss)

    Returns:
        nn.Module: An instantiated loss function.
    """
    loss_name = getattr(config, "loss_func", "RandomCRPSLoss")

    return LossRegistry.get(loss_name, **kwargs)
