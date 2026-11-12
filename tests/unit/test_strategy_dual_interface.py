import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.portfolio.position import PositionSide, PositionView
from quant_backtester.strategies.base import BaseStrategy, Strategy, StrategyMode, StrategySignals
from quant_backtester.strategies.context import StrategyContext
from quant_backtester.strategies.registry import StrategyRegistry


def test_base_strategy_alias():
    assert BaseStrategy is Strategy


def test_strategy_context_history_isolation_and_read_only_position():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=10, freq="15min"),
        "open": [float(100 + i) for i in range(10)],
        "high": [float(105 + i) for i in range(10)],
        "low": [float(95 + i) for i in range(10)],
        "close": [float(101 + i) for i in range(10)],
    })
    pos = PositionView(symbol="XAUUSD", side=PositionSide.FLAT, units=0.0, entry_price=0.0)

    # Context at bar_index = 4 (bars 0..4 closed)
    ctx = StrategyContext(
        time=df.iloc[4]["time"],
        bar_index=4,
        symbol="XAUUSD",
        position=pos,
        data=df,
    )

    assert ctx.bar_index == 4
    assert ctx.symbol == "XAUUSD"

    # PositionView is frozen / read-only
    with pytest.raises(Exception):
        ctx.position.units = 100.0  # type: ignore

    # Context attributes are read-only
    with pytest.raises(AttributeError):
        ctx.bar_index = 9  # type: ignore

    # Request last 3 bars -> indices 2, 3, 4 (closes 103, 104, 105)
    hist3 = ctx.history(3)
    assert len(hist3) == 3
    assert list(hist3["close"]) == [103.0, 104.0, 105.0]

    # Mutating returned history DataFrame must NOT mutate underlying df
    hist3.loc[0, "close"] = 999999.0
    assert df.loc[2, "close"] == 103.0

    # Request more bars than available -> capped at bar_index + 1 (5 bars)
    hist10 = ctx.history(10)
    assert len(hist10) == 5
    assert 106.0 not in hist10["close"].values

    with pytest.raises(ValueError):
        ctx.history(0)
    with pytest.raises(ValueError):
        ctx.history(-2)


def test_mode_validation_rejects_missing_methods():
    class BadVectorized(Strategy):
        name = "bad_vec"
        mode = StrategyMode.VECTORIZED
        # Missing generate_signals override

    class BadEventDriven(Strategy):
        name = "bad_event"
        mode = StrategyMode.EVENT_DRIVEN
        # Missing on_bar override

    class BadBoth(Strategy):
        name = "bad_both"
        mode = StrategyMode.BOTH

        def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
            return StrategySignals(market_data)
        # Missing on_bar override

    with pytest.raises(TypeError, match="generate_signals"):
        StrategyRegistry.register("bad_vec", BadVectorized)

    with pytest.raises(TypeError, match="on_bar"):
        StrategyRegistry.register("bad_event", BadEventDriven)

    with pytest.raises(TypeError, match="on_bar"):
        StrategyRegistry.register("bad_both", BadBoth)


class PureEventDrivenStrategy(Strategy):
    name = "pure_event_driven"
    mode = StrategyMode.EVENT_DRIVEN

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        # Bar 1: emit LONG; Bar 3: emit EXIT_LONG
        if context.bar_index == 1 and context.position.is_flat:
            return Signal(time=context.time, symbol=context.symbol, signal_type=SignalType.LONG)
        if context.bar_index == 3 and context.position.is_long:
            return Signal(time=context.time, symbol=context.symbol, signal_type=SignalType.EXIT_LONG)
        return None


def test_pure_event_driven_execution_timing():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=6, freq="15min"),
        "open": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        "high": [101.0, 103.0, 105.0, 107.0, 109.0, 111.0],
        "low": [99.0, 101.0, 103.0, 105.0, 107.0, 109.0],
        "close": [100.5, 102.5, 104.5, 106.5, 108.5, 110.5],
    })
    df_orig = df.copy(deep=True)
    cfg = BacktestConfig(lot_size=1.0, contract_size=1.0, fee_rate_per_side=0.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    strat = PureEventDrivenStrategy()

    res = BacktestEngine(cfg).run(df, strat)
    # Input DataFrame must not be mutated
    pd.testing.assert_frame_equal(df, df_orig)

    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    # Signal at bar 1 close -> fills at bar 2 open (104.0)
    assert trade["entry_price"] == pytest.approx(104.0)
    assert trade["entry_time"] == pd.Timestamp(df.iloc[2]["time"])
    # Exit signal at bar 3 close -> fills at bar 4 open (108.0)
    assert trade["exit_price"] == pytest.approx(108.0)
    assert trade["exit_time"] == pd.Timestamp(df.iloc[4]["time"])

