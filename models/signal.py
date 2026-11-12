from __future__ import annotations
import copy
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping
import pandas as pd


class SignalType(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1
    EXIT_LONG = 2
    EXIT_SHORT = -2
    EXIT_ALL = 3


def _freeze_mapping(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze_mapping(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze_mapping(v) for v in obj)
    return copy.deepcopy(obj)


@dataclass(frozen=True)
class Signal:
    time: pd.Timestamp
    symbol: str
    signal_type: SignalType
    strength: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        if self.time is None or pd.isna(self.time):
            raise ValueError("Signal.time không được rỗng hoặc NaT.")
        try:
            ts = pd.Timestamp(self.time)
        except Exception as exc:
            raise ValueError(f"Signal.time không hợp lệ: {self.time!r}") from exc
        if pd.isna(ts):
            raise ValueError("Signal.time không được là NaT.")
        object.__setattr__(self, "time", ts)

        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("Signal.symbol phải là chuỗi không rỗng.")

        if not isinstance(self.signal_type, SignalType):
            raise ValueError(f"Signal.signal_type phải thuộc SignalType, nhận {self.signal_type!r}.")

        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise ValueError("Signal.strength phải là số thực hữu hạn.")
        f_strength = float(self.strength)
        if not math.isfinite(f_strength) or not (0.0 <= f_strength <= 1.0):
            raise ValueError("Signal.strength phải là số hữu hạn trong đoạn [0.0, 1.0].")
        object.__setattr__(self, "strength", f_strength)

        raw_meta = self.metadata if self.metadata is not None else {}
        if not isinstance(raw_meta, Mapping):
            raise ValueError("Signal.metadata phải là một dictionary/mapping.")
        object.__setattr__(self, "metadata", _freeze_mapping(raw_meta))

    @property
    def is_entry(self) -> bool:
        return self.signal_type in (SignalType.LONG, SignalType.SHORT)

    @property
    def is_exit(self) -> bool:
        return self.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.EXIT_ALL)
