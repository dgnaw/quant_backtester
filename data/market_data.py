from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from .time_utils import parse_timeframe_delta

VALID_PRICE_TYPES = {"BID", "ASK", "MID"}


@dataclass
class MarketData:
    data: pd.DataFrame
    symbol: str
    timeframe: str
    source: str
    price_type: str = "MID"
    timezone: str = "UTC"
    volume_type: str = "tick_volume"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.symbol = str(self.symbol).strip().upper()
        if not self.symbol:
            raise ValueError("symbol không được để trống.")

        self.timeframe = str(self.timeframe).strip().upper()
        parse_timeframe_delta(self.timeframe)

        self.price_type = str(self.price_type).strip().upper()
        if self.price_type not in VALID_PRICE_TYPES:
            raise ValueError(f"price_type '{self.price_type}' không hợp lệ. Phải thuộc {VALID_PRICE_TYPES}.")

        if self.data.empty:
            raise ValueError("MarketData không được chứa DataFrame rỗng.")

    @property
    def start_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.data["time"].iloc[0])

    @property
    def end_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.data["time"].iloc[-1])

    def __len__(self) -> int:
        return len(self.data)

