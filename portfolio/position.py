from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
import pandas as pd


class PositionSide(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class PositionView:
    symbol: str
    side: PositionSide = PositionSide.FLAT
    units: float = 0.0
    entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None

    @property
    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT or self.units == 0.0

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG and self.units > 0.0

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT and self.units > 0.0



@dataclass
class Position:
    symbol: str
    side: PositionSide = PositionSide.FLAT
    units: float = 0.0
    entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None
    entry_fee: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT or self.units == 0.0

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG and self.units > 0.0

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT and self.units > 0.0

    def open(self, side: PositionSide, price: float, units: float, timestamp: pd.Timestamp, fee: float = 0.0):
        self.side = side
        self.entry_price = float(price)
        self.units = float(units)
        self.entry_time = timestamp
        self.entry_fee = float(fee)

    def unrealized_pnl(self, current_price: float) -> float:
        if self.is_flat:
            return 0.0
        multiplier = 1.0 if self.is_long else -1.0
        return (float(current_price) - self.entry_price) * self.units * multiplier

    def close(self, exit_price: float, timestamp: pd.Timestamp, fee: float = 0.0, reason: str = "EXIT") -> dict[str, Any]:
        if self.is_flat:
            raise ValueError(f"Không thể đóng vị thế đang Flat ({self.symbol}).")

        exit_p = float(exit_price)
        exit_fee_val = float(fee)
        multiplier = 1.0 if self.is_long else -1.0
        gross_pnl = (exit_p - self.entry_price) * self.units * multiplier
        net_pnl = gross_pnl - self.entry_fee - exit_fee_val

        trade_record = {
            "symbol": self.symbol,
            "direction": "LONG" if self.is_long else "SHORT",
            "units": self.units,
            "entry_time": self.entry_time,
            "exit_time": timestamp,
            "entry_price": self.entry_price,
            "exit_price": exit_p,
            "entry_fee": self.entry_fee,
            "exit_fee": exit_fee_val,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "reason": reason,
        }

        # Reset vị thế về Flat
        self.side = PositionSide.FLAT
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_fee = 0.0

        return trade_record

    def to_view(self) -> PositionView:
        return PositionView(
            symbol=self.symbol,
            side=self.side,
            units=self.units,
            entry_price=self.entry_price,
            entry_time=self.entry_time,
        )


