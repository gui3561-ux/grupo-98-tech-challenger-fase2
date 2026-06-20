from collections.abc import Callable
from typing import Any

from torch import nn


class ModelFactory:
    """Factory pattern for creating recommendation models."""

    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a model class by name."""

        def decorator(model_cls: type[nn.Module]) -> type[nn.Module]:
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Instantiate a registered model by name."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            msg = f"Model '{name}' not found. Available: {available}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)

    @classmethod
    def available_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._registry.keys())
