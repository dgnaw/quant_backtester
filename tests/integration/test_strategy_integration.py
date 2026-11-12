from typing import ClassVar
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig, ReversePolicy
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode
from quant_backtester.strategies.context import StrategyContext
from quant_backtester.strategies.parameters import Parameter
from quant_backtester.strategies.registry import StrategyRegistry
from quant_backtester.walk_forward import walk_forward_mean_reversion


class CustomBreakoutWithDirectivesStrategy(Strategy):
    """
    Chiến lược mới hoàn toàn (Event-driven kèm Directives SL/TP và Quantity)
    được thêm vào và đăng ký qua StrategyRegistry mà không cần chỉnh sửa bất kỳ dòng code nào của Engine.
    """
    name = "custom_breakout_directives"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN
    parameters: ClassVar[tuple[Parameter, ...]] = (
        Parameter(name="lookback", param_type=int, default=3, min_value=2, max_value=50),
        Parameter(name="order_lots", param_type=float, default=0.05, min_value=0.01, max_value=10.0),
    )

    @property
    def warmup_period(self) -> int:
        return int(self.lookback)

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        hist = context.history(self.lookback)
        if len(hist) < self.lookback:
            return None

        prev_high = float(hist.iloc[:-1]["high"].max())
        close_p = float(bar["close"])

        if context.position.is_flat and close_p > prev_high:
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={
                    "directives": {
                        "quantity": self.order_lots,
                        "stop_loss_price": close_p - 15.0,
                        "take_profit_price": close_p + 30.0,
                    }
                },
            )
        return None


def test_end_to_end_custom_strategy_with_registry_and_directives():
    StrategyRegistry.register("custom_breakout", CustomBreakoutWithDirectivesStrategy)
    try:
        strat = StrategyRegistry.get("custom_breakout", lookback=3, order_lots=0.05)
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01 00:00", periods=8, freq="15min"),
            "open":  [2000.0, 2001.0, 2002.0, 2003.0, 2010.0, 2020.0, 2035.0, 2040.0],
            "high":  [2005.0, 2005.0, 2005.0, 2012.0, 2025.0, 2045.0, 2042.0, 2045.0],
            "low":   [1995.0, 1996.0, 1997.0, 2000.0, 2008.0, 2018.0, 2030.0, 2035.0],
            "close": [2001.0, 2002.0, 2003.0, 2010.0, 2020.0, 2040.0, 2038.0, 2042.0],
        })
        # Bar index 3 (4th bar, after warmup=3): close=2010 > prev_high(2005) -> emits LONG with TP=2040.0, lots=0.05
        # Bar index 4 Open = 2010.0 -> Fills BUY 0.05 lots (5.0 units)
        # Bar index 5 High = 2045.0 >= TP(2040.0) -> Exits at TP = 2040.0
        cfg = BacktestConfig(fee_rate_per_side=0.0, slippage_price=0.0)
        res = BacktestEngine(cfg).run(df, strat)

        assert res.effective_trade_start == 3
        assert len(res.trades) >= 1
        first_trade = res.trades.iloc[0]
        assert first_trade["units"] == pytest.approx(5.0)  # 0.05 lots * 100 contract_size
        assert first_trade["entry_price"] == pytest.approx(2010.0)
        assert first_trade["exit_price"] == pytest.approx(2040.0)
        assert first_trade["reason"] == "TAKE_PROFIT"
    finally:
        StrategyRegistry.unregister("custom_breakout")


def test_walk_forward_uses_warmup_period_and_isolated_instances():
    n = 150
    closes = 2000.0 + pd.Series(range(n)).apply(lambda x: (x % 10) - 5.0).values
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=n, freq="15min"),
        "open": closes,
        "high": closes + 2.0,
        "low": closes - 2.0,
        "close": closes,
    })
    cfg = BacktestConfig()
    folds_df, summary_df = walk_forward_mean_reversion(df, cfg, windows=(10, 15), multipliers=(1.5,), folds=3)
    assert isinstance(folds_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)

