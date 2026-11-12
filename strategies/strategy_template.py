"""
Template mẫu chuẩn V1.0 để triển khai một Strategy mới trong quant_backtester.

Hướng dẫn tạo Strategy mới:
1. Sao chép file này sang một file mới trong thư mục `strategies/` (ví dụ: `trend_following.py`).
2. Chọn `mode`:
   - `StrategyMode.VECTORIZED`: chỉ cần override `generate_signals(market_data)`.
   - `StrategyMode.EVENT_DRIVEN`: chỉ cần override `on_bar(bar, context)`.
   - `StrategyMode.BOTH`: override cả hai phương thức.
3. Khai báo `parameters: ClassVar[tuple[Parameter, ...]]` để tự động validate tham số và ép kiểu CLI.
4. Khai báo `@property def warmup_period(self) -> int` để Engine tự động mask tín hiệu trước khi đủ nến.
5. Đăng ký Strategy vào `StrategyRegistry.register("ten_chien_luoc", TenClass)`.
"""
from __future__ import annotations
from typing import Any, ClassVar
import pandas as pd

from quant_backtester.models.signal import Signal, SignalType
from .base import BaseStrategy, Strategy, StrategyMode, StrategySignals
from .context import FillView, StrategyContext
from .parameters import Parameter


class NewStrategy(Strategy):
    name = "new_strategy"
    mode: ClassVar[StrategyMode] = StrategyMode.BOTH
    parameters: ClassVar[tuple[Parameter, ...]] = (
        Parameter(
            name="example_parameter",
            param_type=int,
            default=14,
            min_value=1,
            max_value=500,
            step=1,
            optimizable=True,
            description="Chu kỳ mẫu cho chỉ báo",
        ),
    )

    def __init__(self, example_parameter: int = 14, **kwargs: Any):
        super().__init__(example_parameter=example_parameter, **kwargs)

    @property
    def warmup_period(self) -> int:
        return int(self.example_parameter)

    def _initial_state(self) -> dict[str, Any]:
        return {"bars_processed": 0}

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = 0          # +1 Long, -1 Short, 0 no entry
        df["exit_long_signal"] = False  # close an existing Long
        df["exit_short_signal"] = False # close an existing Short
        return StrategySignals(df)

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        self.state["bars_processed"] += 1
        hist = context.history(self.example_parameter)
        if len(hist) < self.example_parameter:
            return None
        return Signal(
            time=context.time,
            symbol=context.symbol,
            signal_type=SignalType.FLAT,
        )

    def on_fill(self, fill_view: FillView) -> None:
        pass
