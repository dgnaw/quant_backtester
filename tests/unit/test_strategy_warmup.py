from typing import ClassVar
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.strategies.parameters import Parameter


class EarlySignalStrategy(Strategy):
    """Strategy cố tình phát tín hiệu ở mọi nến (kể cả trong giai đoạn warm-up)."""
    name = "early_signal_strategy"
    parameters: ClassVar[tuple[Parameter, ...]] = (
        Parameter(name="warmup", param_type=int, default=5, min_value=0),
    )

    @property
    def warmup_period(self) -> int:
        return self.warmup

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = 1  # Cố tình phát LONG tại mọi nến từ index 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)


class InvalidWarmupStrategy(Strategy):
    name = "invalid_warmup"

    @property
    def warmup_period(self) -> int:
        return -3

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)


def _make_bars(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=n, freq="15min"),
        "open": [100.0 + i for i in range(n)],
        "high": [105.0 + i for i in range(n)],
        "low": [95.0 + i for i in range(n)],
        "close": [101.0 + i for i in range(n)],
    })


def test_warmup_masks_early_signals_and_fills_at_index_n():
    # warmup_period = 5 -> nến index 4 (5-1) là nến đầu tiên đủ điều kiện phát Signal,
    # và nến index 5 là nến đầu tiên thực hiện Fill tại Open.
    df = _make_bars(10)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0, fee_rate_per_side=0.0)
    strat = EarlySignalStrategy(warmup=5)
    engine = BacktestEngine(cfg)

    res = engine.run(df, strat, trade_start=0)
    assert res.effective_trade_start == 5
    assert len(res.trades) == 1
    first_trade = res.trades.iloc[0]
    # Khớp tại Open của nến index 5 (open = 100.0 + 5 = 105.0)
    assert first_trade["entry_price"] == pytest.approx(105.0)
    assert first_trade["entry_time"] == pd.Timestamp(df.iloc[5]["time"])


def test_effective_trade_start_clamping():
    df = _make_bars(12)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0, fee_rate_per_side=0.0)
    strat = EarlySignalStrategy(warmup=5)
    engine = BacktestEngine(cfg)

    # Khi trade_start = 2 < warmup = 5 -> effective_trade_start = 5
    res1 = engine.run(df, strat, trade_start=2)
    assert res1.effective_trade_start == 5

    # Khi trade_start = 8 > warmup = 5 -> effective_trade_start = 8
    res2 = engine.run(df, strat, trade_start=8)
    assert res2.effective_trade_start == 8
    assert res2.trades.iloc[0]["entry_price"] == pytest.approx(108.0)


def test_sma20_signal_at_bar_19_fills_at_bar_20():
    # Tạo 25 nến: 19 nến đầu giá 100, nến index 19 (nến thứ 20) giá sập mạnh xuống 50 để thủng lower band
    n = 25
    df = _make_bars(n)
    df["open"] = 100.0
    df["high"] = 101.0
    df["low"] = 99.0
    df["close"] = 100.0
    df.loc[19, "low"] = 49.0
    df.loc[19, "close"] = 50.0  # Bar index 19 (20th bar) triggers Bollinger lower band
    df.loc[20, "open"] = 55.0   # Bar index 20 Open price where fill must happen
    df.loc[20, "low"] = 54.0

    strat = MeanReversionStrategy(window=20, std_multiplier=2.0)
    assert strat.warmup_period == 20

    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0, fee_rate_per_side=0.0)
    res = BacktestEngine(cfg).run(df, strat)
    assert res.effective_trade_start == 20
    assert len(res.trades) >= 1
    assert res.trades.iloc[0]["entry_time"] == pd.Timestamp(df.iloc[20]["time"])
    assert res.trades.iloc[0]["entry_price"] == pytest.approx(55.0)


def test_warmup_validation_and_insufficient_data():
    df = _make_bars(10)
    engine = BacktestEngine()

    with pytest.raises(ValueError, match="warmup_period"):
        engine.run(df, InvalidWarmupStrategy())

    # Không đủ dữ liệu cho warmup_period = 20 khi chỉ có 15 nến
    strat20 = MeanReversionStrategy(window=20, std_multiplier=2.0)
    with pytest.raises(ValueError, match="dữ liệu"):
        engine.run(_make_bars(15), strat20)

    # Đủ đúng 20 nến để tính chỉ báo tại bar 19 nhưng không có bar 20 để Fill -> hoàn thành an toàn với 0 lệnh
    res_exact = engine.run(_make_bars(20), strat20)
    assert len(res_exact.trades) == 0
    assert res_exact.effective_trade_start == 20

