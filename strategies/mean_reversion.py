from __future__ import annotations
from typing import ClassVar
import pandas as pd

from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.portfolio.position import PositionSide
from .base import Strategy, StrategyMode, StrategySignals
from .context import StrategyContext
from .parameters import Parameter


class MeanReversionStrategy(Strategy):
    """Bollinger mean reversion: fade outer-band excursions, exit at middle band."""
    name = "bollinger_mean_reversion"
    mode: ClassVar[StrategyMode] = StrategyMode.BOTH
    parameters: ClassVar[tuple[Parameter, ...]] = (
        Parameter(
            name="window",
            param_type=int,
            default=20,
            min_value=2,
            max_value=500,
            step=1,
            optimizable=True,
            description="Chu kỳ tính SMA và độ lệch chuẩn Bollinger Bands",
        ),
        Parameter(
            name="std_multiplier",
            param_type=float,
            default=2.0,
            min_value=0.0001,
            max_value=20.0,
            step=0.1,
            optimizable=True,
            description="Hệ số nhân độ lệch chuẩn Bollinger Bands",
        ),
    )

    def __init__(self, window: int = 20, std_multiplier: float = 2.0, **kwargs):
        super().__init__(window=window, std_multiplier=std_multiplier, **kwargs)

    @property
    def warmup_period(self) -> int:
        return int(self.window)

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["sma"] = df["close"].rolling(self.window, min_periods=self.window).mean()
        df["std"] = df["close"].rolling(self.window, min_periods=self.window).std(ddof=1)
        df["upper_band"] = df["sma"] + self.std_multiplier * df["std"]
        df["lower_band"] = df["sma"] - self.std_multiplier * df["std"]
        df["entry_signal"] = 0
        df.loc[df["close"] < df["lower_band"], "entry_signal"] = 1
        df.loc[df["close"] > df["upper_band"], "entry_signal"] = -1
        # Direction-specific exits prevent a long rule from closing a short (and vice versa).
        df["exit_long_signal"] = df["sma"].notna() & (df["close"] >= df["sma"])
        df["exit_short_signal"] = df["sma"].notna() & (df["close"] <= df["sma"])
        return StrategySignals(df)

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        hist = context.history(self.window)
        if len(hist) < self.window:
            return None

        closes = hist["close"].astype(float)
        sma = float(closes.mean())
        std = float(closes.std(ddof=1))
        upper_band = sma + self.std_multiplier * std
        lower_band = sma - self.std_multiplier * std
        close_price = float(bar["close"])
        ts = pd.Timestamp(bar["time"])

        pos = context.position
        if pos.side == PositionSide.LONG:
            if close_price >= sma:
                return Signal(time=ts, symbol=context.symbol, signal_type=SignalType.EXIT_LONG)
            return None
        if pos.side == PositionSide.SHORT:
            if close_price <= sma:
                return Signal(time=ts, symbol=context.symbol, signal_type=SignalType.EXIT_SHORT)
            return None

        # Khi đang FLAT
        if close_price < lower_band:
            return Signal(time=ts, symbol=context.symbol, signal_type=SignalType.LONG)
        if close_price > upper_band:
            return Signal(time=ts, symbol=context.symbol, signal_type=SignalType.SHORT)
        return None
