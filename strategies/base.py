from __future__ import annotations
from abc import ABC
import copy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar
import pandas as pd

from .parameters import Parameter, ParameterSchema

if TYPE_CHECKING:
    from quant_backtester.models.signal import Signal
    from .context import FillView, StrategyContext


class StrategyMode(Enum):
    VECTORIZED = "VECTORIZED"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    BOTH = "BOTH"


@dataclass
class StrategySignals:
    """Signals known at candle close and executed by the engine at next candle open."""
    data: pd.DataFrame


class Strategy(ABC):
    name: str = "base_strategy"
    mode: ClassVar[StrategyMode] = StrategyMode.VECTORIZED
    parameters: ClassVar[tuple[Parameter, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Kiểm tra mode hợp lệ khi định nghĩa subclass
        if not isinstance(getattr(cls, "mode", None), StrategyMode):
            raise TypeError(f"Strategy '{cls.__name__}' phải khai báo mode thuộc StrategyMode.")

    def __init__(self, *args: Any, **kwargs: Any):
        schema_params = getattr(self.__class__, "parameters", ())
        if schema_params:
            merged_kwargs = dict(kwargs)
            for idx, val in enumerate(args):
                if idx >= len(schema_params):
                    raise TypeError(f"{self.__class__.__name__} nhận quá nhiều đối số vị trí.")
                param_name = schema_params[idx].name
                if param_name in merged_kwargs:
                    raise TypeError(f"Tham số '{param_name}' được truyền lặp lại.")
                merged_kwargs[param_name] = val

            validated = self.validate_parameters(merged_kwargs)
            object.__setattr__(self, "_parameters", dict(validated))
            for k, v in validated.items():
                object.__setattr__(self, k, v)
            object.__setattr__(self, "_params_locked", True)
        else:
            object.__setattr__(self, "_parameters", dict(kwargs))
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)
            object.__setattr__(self, "_params_locked", False)

        object.__setattr__(self, "_state", copy.deepcopy(self._initial_state()))

    def __setattr__(self, key: str, value: Any) -> None:
        if getattr(self, "_params_locked", False):
            params_dict = getattr(self, "_parameters", {})
            if key in params_dict or key == "_parameters":
                raise AttributeError(
                    f"Tham số '{key}' của Strategy là bất biến sau khi khởi tạo. Hãy tạo instance mới nếu cần đổi tham số."
                )
        object.__setattr__(self, key, value)

    @classmethod
    def validate_parameters(cls, raw_params: dict[str, Any] | None = None, coerce: bool = False) -> dict[str, Any]:
        schema = ParameterSchema(getattr(cls, "parameters", ()))
        return schema.validate(raw_params, coerce=coerce)

    @classmethod
    def validate_mode_contract(cls) -> None:
        mode = getattr(cls, "mode", StrategyMode.VECTORIZED)
        if not isinstance(mode, StrategyMode):
            raise TypeError(f"Mode của '{cls.__name__}' phải thuộc StrategyMode.")

        has_vec = cls.generate_signals is not Strategy.generate_signals
        has_event = cls.on_bar is not Strategy.on_bar

        if mode == StrategyMode.VECTORIZED and not has_vec:
            raise TypeError(f"Strategy '{cls.__name__}' khai báo VECTORIZED nhưng chưa override generate_signals().")
        if mode == StrategyMode.EVENT_DRIVEN and not has_event:
            raise TypeError(f"Strategy '{cls.__name__}' khai báo EVENT_DRIVEN nhưng chưa override on_bar().")
        if mode == StrategyMode.BOTH and not (has_vec and has_event):
            raise TypeError(f"Strategy '{cls.__name__}' khai báo BOTH nhưng chưa override đủ cả generate_signals() và on_bar().")

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        """Return data with entry_signal (-1 short, 0 none, +1 long), exit_long_signal and exit_short_signal."""
        raise NotImplementedError(f"{self.__class__.__name__} không triển khai generate_signals().")

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | list[Signal] | None:
        """Process a single closed candle in event-driven mode."""
        raise NotImplementedError(f"{self.__class__.__name__} không triển khai on_bar().")

    def get_parameters(self) -> dict[str, Any]:
        return copy.deepcopy(getattr(self, "_parameters", {}))

    def _initial_state(self) -> dict[str, Any]:
        return {}

    def reset(self) -> None:
        object.__setattr__(self, "_state", copy.deepcopy(self._initial_state()))

    @property
    def state(self) -> dict[str, Any]:
        if not hasattr(self, "_state"):
            object.__setattr__(self, "_state", copy.deepcopy(self._initial_state()))
        return self._state

    @property
    def warmup_period(self) -> int:
        return 0

    def on_fill(self, fill_view: FillView) -> None:
        """Optional hook called with a read-only FillView when an order is filled."""
        pass

    def on_intent_rejected(self, intent: Any, reason: str) -> None:
        """Optional hook called when RiskManager rejects an OrderIntent."""
        pass


BaseStrategy = Strategy
