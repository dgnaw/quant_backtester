from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd

from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.portfolio.position import PositionView


@dataclass(frozen=True)
class FillView:
    """View chỉ đọc của sự kiện Fill truyền cho hook on_fill() của Strategy."""
    fill_id: str
    order_id: str
    time: pd.Timestamp
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    units: float
    fee: float
    slippage: float
    reason: str

    @classmethod
    def from_fill(cls, fill: Fill) -> FillView:
        return cls(
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            time=fill.time,
            symbol=fill.symbol,
            side=fill.side,
            price=fill.price,
            quantity=fill.quantity,
            units=fill.units,
            fee=fill.fee,
            slippage=fill.slippage,
            reason=fill.reason,
        )


class StrategyContext:
    """
    Context chỉ đọc được truyền vào Strategy.on_bar(bar, context).
    Không cung cấp Account, Balance, Equity, PnL hay quyền sửa DataFeed gốc.
    Đảm bảo chống Lookahead Bias: history(n) chỉ trả về các nến từ 0 đến bar_index hiện tại (đã đóng).
    """

    def __init__(
        self,
        time: pd.Timestamp,
        bar_index: int,
        symbol: str,
        position: PositionView,
        data: pd.DataFrame,
    ):
        self._time = pd.Timestamp(time)
        self._bar_index = int(bar_index)
        self._symbol = str(symbol)
        self._position = position
        self._data = data

    @property
    def time(self) -> pd.Timestamp:
        return self._time

    @property
    def bar_index(self) -> int:
        return self._bar_index

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def position(self) -> PositionView:
        return self._position

    def history(self, n: int) -> pd.DataFrame:
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("Số lượng nến lịch sử n phải là số nguyên dương (> 0).")
        end_idx = self._bar_index + 1
        start_idx = max(0, end_idx - n)
        # Trả về bản sao (copy) để Strategy không thể sửa đổi dữ liệu gốc trong Engine/DataFeed
        return self._data.iloc[start_idx:end_idx].copy().reset_index(drop=True)

