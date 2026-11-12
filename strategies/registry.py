from __future__ import annotations
from typing import Type
from .base import Strategy


class StrategyRegistry:
    _registry: dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[Strategy] | None = None):
        """Đăng ký strategy class hoặc dùng như decorator."""
        if strategy_cls is not None:
            if not issubclass(strategy_cls, Strategy):
                raise TypeError(f"'{strategy_cls}' phải kế thừa từ Strategy.")
            strategy_cls.validate_mode_contract()
            cls._registry[name] = strategy_cls
            return strategy_cls

        def decorator(subclass: Type[Strategy]):
            if not issubclass(subclass, Strategy):
                raise TypeError(f"'{subclass}' phải kế thừa từ Strategy.")
            subclass.validate_mode_contract()
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        """Gỡ bỏ chiến lược khỏi Registry để phục vụ kiểm thử và cô lập trạng thái."""
        cls._registry.pop(name, None)

    @classmethod
    def get(cls, name: str, **kwargs) -> Strategy:
        if name not in cls._registry:
            raise KeyError(
                f"Chiến lược '{name}' không tồn tại trong Registry. Các chiến lược hiện có: {cls.list_strategies()}"
            )
        strategy_cls = cls._registry[name]
        if getattr(strategy_cls, "parameters", None):
            validated_kwargs = strategy_cls.validate_parameters(kwargs)
            return strategy_cls(**validated_kwargs)
        return strategy_cls(**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        return sorted(list(cls._registry.keys()))
