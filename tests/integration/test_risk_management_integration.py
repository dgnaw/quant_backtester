"""
End-to-End Integration tests for Group F Risk Management & Portfolio Risk Control.
Covers all 8 required integration scenarios with BacktestEngine.
"""

from __future__ import annotations

import math
from typing import Any
import pandas as pd

from quant_backtester.config import (
    AccountConfig,
    CommissionMode,
    EngineConfig,
    FeeConfig,
    InstrumentConfig,
    PositionMode,
    ReversePolicy,
    RiskConfig,
    SizingBasis,
    SizingMode,
    SlippageConfig,
    SpreadConfig,
)
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.order import OrderStatus
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode


def _sig(
    time: pd.Timestamp,
    symbol: str,
    signal_type: SignalType,
    quantity: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
) -> Signal:
    directives: dict[str, Any] = {}
    if quantity is not None:
        directives["quantity"] = quantity
    if stop_loss_price is not None:
        directives["stop_loss_price"] = stop_loss_price
    if take_profit_price is not None:
        directives["take_profit_price"] = take_profit_price
    meta = {"directives": directives} if directives else None
    return Signal(time=time, symbol=symbol, signal_type=signal_type, metadata=meta)


class ScriptedEventStrategy(Strategy):
    """Event-driven strategy emitting scripted signals by bar index and recording rejections."""

    mode = StrategyMode.EVENT_DRIVEN
    warmup_period = 0

    def __init__(self, signals_by_bar: dict[int, Signal]):
        super().__init__()
        self.signals_by_bar = signals_by_bar
        self.rejections: list[tuple[int, str]] = []
        self._current_bar_idx = 0

    def reset(self) -> None:
        self.rejections.clear()
        self._current_bar_idx = 0

    def on_bar(self, bar: pd.Series, context) -> Signal | None:
        self._current_bar_idx = context.bar_index
        return self.signals_by_bar.get(context.bar_index)

    def on_intent_rejected(self, intent_or_order, reason: str) -> None:
        self.rejections.append((self._current_bar_idx, reason))


def _make_bars(prices: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for t_str, o, h, l, c in prices:
        rows.append(
            {
                "time": pd.Timestamp(t_str, tz="UTC"),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_scenario_1_risk_based_sizing_end_to_end_in_engine():
    """
    Scenario 1: Risk-based sizing in BacktestEngine:
    Account $10,000, risk_per_trade_pct = 1% ($100).
    Signal BUY at bar 0 close (2000.0) with stop_loss_price=1990.0 -> sized to 0.10 lot (10 oz).
    Fills at bar 1 open (2000.0), hits SL at bar 2 low (1988.0 -> fills at 1990.0), realized loss = -$100.0!
    """
    bars = _make_bars(
        [
            ("2025-01-01 00:00:00", 2000, 2002, 1998, 2000),
            ("2025-01-01 01:00:00", 2000, 2005, 1995, 2002),
            ("2025-01-01 02:00:00", 2000, 2001, 1988, 1992),
            ("2025-01-01 03:00:00", 1992, 1995, 1990, 1994),
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=1.0),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=0.0),
        slippage=SlippageConfig(slippage_rate=0.0, slippage_price=0.0),
        spread=SpreadConfig(spread_price=0.0),
        risk=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            sizing_basis=SizingBasis.EQUITY,
            risk_per_trade_pct=0.01,
        ),
    )
    strat = ScriptedEventStrategy(
        {
            0: _sig(
                time=bars.loc[0, "time"],
                symbol="XAUUSD",
                signal_type=SignalType.LONG,
                stop_loss_price=1990.0,
            )
        }
    )
    engine = BacktestEngine(config=cfg)
    result = engine.run(bars, strat)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert math.isclose(float(trade["quantity"]), 0.10)
    assert math.isclose(float(trade["net_pnl"]), -100.0)
    assert math.isclose(result.final_balance, 9900.0)


def test_scenario_2_per_trade_risk_cap_rejects_oversized_fixed_lot_in_engine():
    """
    Scenario 2: Fixed lot order exceeding max_risk_per_trade_amount is rejected by RiskManager,
    recorded as REJECTED in result.orders and result.risk_events.
    """
    bars = _make_bars(
        [
            ("2025-01-01 00:00:00", 2000, 2002, 1998, 2000),
            ("2025-01-01 01:00:00", 2000, 2005, 1995, 2002),
            ("2025-01-01 02:00:00", 2002, 2005, 2000, 2003),
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.50),
        fee=FeeConfig(fee_rate_per_side=0.0),
        risk=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_amount=200.0,
        ),
    )
    strat = ScriptedEventStrategy(
        {
            0: _sig(
                time=bars.loc[0, "time"],
                symbol="XAUUSD",
                signal_type=SignalType.LONG,
                quantity=0.50,
                stop_loss_price=1990.0,
            )
        }
    )
    result = BacktestEngine(config=cfg).run(bars, strat)
    assert len(result.trades) == 0
    assert len(result.orders) == 1
    assert result.orders.iloc[0]["status"] == OrderStatus.REJECTED.value
    assert result.orders.iloc[0]["reject_reason"] == "MAX_TRADE_RISK_EXCEEDED"
    assert strat.rejections == [(0, "MAX_TRADE_RISK_EXCEEDED")]
    assert "ORDER_REJECTED" in set(result.risk_events["event_type"])


def test_scenario_3_daily_loss_limit_blocks_same_day_and_resumes_next_day():
    """
    Scenario 3: Trade 1 on Day 1 loses $300, breaching max_daily_loss_amount=$250.
    Signal 2 on Day 1 is rejected with DAILY_LOSS_LIMIT_REACHED.
    Signal 3 on Day 2 is approved and executed after daily loss reset!
    """
    bars = _make_bars(
        [
            ("2025-01-02 09:00:00", 2000, 2001, 1999, 2000),  # Bar 0: emit BUY #1
            ("2025-01-02 10:00:00", 2000, 2001, 1965, 1970),  # Bar 1: fill @ 2000, SL hit @ 1970 (-$300 on 0.1 lot) -> emit BUY #2 at close (rejected!)
            ("2025-01-02 11:00:00", 1970, 1975, 1968, 1972),  # Bar 2: still Day 1
            ("2025-01-03 09:00:00", 1972, 1975, 1970, 1974),  # Bar 3: Day 2! emit BUY #3 (approved!)
            ("2025-01-03 10:00:00", 1974, 1986, 1973, 1984),  # Bar 4: fill BUY #3 @ 1974, closes at end @ 1984 (+$100)
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10),
        fee=FeeConfig(fee_rate_per_side=0.0),
        risk=RiskConfig(max_daily_loss_amount=250.0),
    )
    strat = ScriptedEventStrategy(
        {
            0: _sig(time=bars.loc[0, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, stop_loss_price=1970.0),
            1: _sig(time=bars.loc[1, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, stop_loss_price=1950.0),
            3: _sig(time=bars.loc[3, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, stop_loss_price=1950.0),
        }
    )
    result = BacktestEngine(config=cfg).run(bars, strat)

    assert len(result.trades) == 2
    assert strat.rejections == [(1, "DAILY_LOSS_LIMIT_REACHED")]
    assert math.isclose(result.final_balance, 9800.0)


def test_scenario_4_max_drawdown_circuit_breaker_with_auto_liquidation():
    """
    Scenario 4: Max Drawdown circuit breaker with liquidate_on_max_drawdown=True:
    Position opens at 2000 (0.50 lot = 50 oz) with wide SL.
    At bar 2 close, price drops to 1975 -> floating loss = -$1,250 (12.5% drawdown >= 10% limit).
    Engine automatically liquidates the position at bar 2 close with reason='MAX_DRAWDOWN_LIQUIDATION'
    and blocks subsequent entry signals even on Day 2!
    """
    bars = _make_bars(
        [
            ("2025-01-02 09:00:00", 2000, 2002, 1998, 2000),  # Bar 0: emit BUY
            ("2025-01-02 10:00:00", 2000, 2002, 1995, 1996),  # Bar 1: fill @ 2000
            ("2025-01-02 11:00:00", 1996, 1996, 1974, 1975),  # Bar 2: close=1975 -> DD=12.5% -> auto liquidate!
            ("2025-01-03 09:00:00", 1975, 1980, 1970, 1978),  # Bar 3: Day 2 emit BUY -> rejected!
            ("2025-01-03 10:00:00", 1978, 1985, 1975, 1980),
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.50),
        fee=FeeConfig(fee_rate_per_side=0.0),
        risk=RiskConfig(
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            max_drawdown_pct=0.10,
            liquidate_on_max_drawdown=True,
        ),
    )
    strat = ScriptedEventStrategy(
        {
            0: _sig(time=bars.loc[0, "time"], symbol="XAUUSD", signal_type=SignalType.LONG),
            3: _sig(time=bars.loc[3, "time"], symbol="XAUUSD", signal_type=SignalType.LONG),
        }
    )
    result = BacktestEngine(config=cfg).run(bars, strat)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["reason"] == "MAX_DRAWDOWN_LIQUIDATION"
    assert math.isclose(float(result.trades.iloc[0]["net_pnl"]), -1250.0)
    assert strat.rejections == [(3, "MAX_DRAWDOWN_LIMIT_REACHED")]


def test_scenario_5_margin_stop_out_liquidates_losing_position_and_preserves_invariants():
    """
    Scenario 5: Margin Stop-Out:
    Initial balance = $2,500, Leverage = 100:1, stop_out_level_pct = 50%.
    Open 1.0 lot XAUUSD @ 2000 (used_margin = $2,000).
    At bar 2 close, price drops to 1984 -> floating PnL = -$1,600 -> equity = $900, used_margin = $1,984
    -> margin_level_pct = 45.36% <= 50% -> Forced Stop-Out liquidation!
    """
    bars = _make_bars(
        [
            ("2025-01-02 09:00:00", 2000, 2001, 1999, 2000),  # Bar 0: emit BUY 1.0 lot
            ("2025-01-02 10:00:00", 2000, 2002, 1995, 1995),  # Bar 1: fill @ 2000
            ("2025-01-02 11:00:00", 1995, 1995, 1983, 1984),  # Bar 2: close @ 1984 -> Stop-Out!
            ("2025-01-02 12:00:00", 1984, 1990, 1980, 1988),
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=2500.0, leverage=100.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=1.0),
        fee=FeeConfig(fee_rate_per_side=0.0),
        risk=RiskConfig(
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            leverage=100.0,
            enforce_margin_check=True,
            margin_call_level_pct=100.0,
            stop_out_level_pct=50.0,
        ),
    )
    strat = ScriptedEventStrategy(
        {
            0: _sig(time=bars.loc[0, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, quantity=1.0),
        }
    )
    result = BacktestEngine(config=cfg).run(bars, strat)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["reason"] == "MARGIN_STOP_OUT"
    assert math.isclose(result.final_balance, 900.0)
    assert "MARGIN_STOP_OUT" in set(result.risk_events["event_type"])


def test_scenario_6_scale_in_blocked_by_max_symbol_lots():
    """
    Scenario 6: In NETTING_SCALE_IN mode, first entry of 0.60 lot succeeds,
    second scale-in of 0.60 lot is rejected by max_symbol_lots=1.0,
    while subsequent exit signal still closes the 0.60 lot position cleanly.
    """
    bars = _make_bars(
        [
            ("2025-01-02 09:00:00", 2000, 2002, 1998, 2000),  # Bar 0: emit BUY 0.60
            ("2025-01-02 10:00:00", 2000, 2005, 1999, 2004),  # Bar 1: fill 0.60; emit scale-in BUY 0.60 -> rejected!
            ("2025-01-02 11:00:00", 2004, 2010, 2002, 2008),  # Bar 2: emit EXIT_LONG
            ("2025-01-02 12:00:00", 2010, 2012, 2008, 2010),  # Bar 3: exit fills @ 2010
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.60),
        default_position_mode=PositionMode.NETTING_SCALE_IN,
        fee=FeeConfig(fee_rate_per_side=0.0),
        risk=RiskConfig(stop_loss_pct=0.0, take_profit_pct=0.0, max_symbol_lots=1.0),
    )
    strat = ScriptedEventStrategy(
        {
            0: _sig(time=bars.loc[0, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, quantity=0.60),
            1: _sig(time=bars.loc[1, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, quantity=0.60),
            2: _sig(time=bars.loc[2, "time"], symbol="XAUUSD", signal_type=SignalType.EXIT_LONG),
        }
    )
    result = BacktestEngine(config=cfg).run(bars, strat)

    assert strat.rejections == [(1, "SYMBOL_LOT_LIMIT_EXCEEDED")]
    assert len(result.trades) == 1
    assert math.isclose(float(result.trades.iloc[0]["quantity"]), 0.60)
    assert math.isclose(float(result.trades.iloc[0]["net_pnl"]), 600.0)


def test_scenario_7_two_order_reverse_when_halted_closes_old_position_and_rejects_new_entry():
    """
    Scenario 7: Under ReversePolicy.TWO_ORDER_REVERSE, when position has a large unrealized loss
    that triggers HALTED_DAILY_LOSS at bar 1 close, emitting an opposite SHORT signal at bar 1 close
    creates the STRATEGY_EXIT order (which fills at bar 2 open and closes the LONG position!),
    and then the reverse entry order at bar 2 open is rejected by DAILY_LOSS_LIMIT_REACHED!
    """
    bars = _make_bars(
        [
            ("2025-01-02 09:00:00", 2000, 2002, 1998, 2000),  # Bar 0: emit LONG
            ("2025-01-02 10:00:00", 2000, 2001, 1965, 1968),  # Bar 1: fill LONG @ 2000; close=1968 (-$320 loss >= $250 limit -> HALT!); emit SHORT reverse
            ("2025-01-02 11:00:00", 1968, 1972, 1960, 1965),  # Bar 2: open closes LONG @ 1968; reverse SHORT entry is rejected!
            ("2025-01-02 12:00:00", 1965, 1970, 1960, 1965),
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10),
        reverse_policy=ReversePolicy.TWO_ORDER_REVERSE,
        fee=FeeConfig(fee_rate_per_side=0.0),
        risk=RiskConfig(stop_loss_pct=0.0, take_profit_pct=0.0, max_daily_loss_amount=250.0),
    )
    strat = ScriptedEventStrategy(
        {
            0: _sig(time=bars.loc[0, "time"], symbol="XAUUSD", signal_type=SignalType.LONG),
            1: _sig(time=bars.loc[1, "time"], symbol="XAUUSD", signal_type=SignalType.SHORT),
        }
    )
    result = BacktestEngine(config=cfg).run(bars, strat)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["direction"] == "LONG"
    assert math.isclose(float(result.trades.iloc[0]["net_pnl"]), -320.0)
    rejected_orders = result.orders[result.orders["status"] == OrderStatus.REJECTED.value]
    assert len(rejected_orders) == 1
    assert pd.Timestamp(rejected_orders.iloc[0]["time"]) == pd.Timestamp(bars.loc[2, "time"])
    assert rejected_orders.iloc[0]["reject_reason"] == "DAILY_LOSS_LIMIT_REACHED"
    assert [reason for _, reason in strat.rejections] == ["DAILY_LOSS_LIMIT_REACHED"]
    assert result.open_positions.empty


def test_scenario_8_determinism_across_repeated_engine_runs():
    """
    Scenario 8: Running the same BacktestEngine twice with complex RiskConfig produces
    bit-for-bit identical trades, orders, equity, ledger, and risk_events DataFrames.
    """
    bars = _make_bars(
        [
            ("2025-01-02 09:00:00", 2000, 2002, 1998, 2000),
            ("2025-01-02 10:00:00", 2000, 2015, 1995, 2010),
            ("2025-01-02 11:00:00", 2010, 2012, 1985, 1988),
            ("2025-01-03 09:00:00", 1988, 2005, 1985, 2002),
            ("2025-01-03 10:00:00", 2002, 2020, 2000, 2015),
        ]
    )
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0, leverage=100.0),
        instrument=InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10),
        fee=FeeConfig(commission_per_lot=5.0, commission_mode=CommissionMode.PER_SIDE),
        slippage=SlippageConfig(slippage_price=0.10),
        risk=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_pct=0.01,
            max_daily_loss_amount=300.0,
            enforce_margin_check=True,
            leverage=100.0,
        ),
    )
    signals = {
        0: _sig(time=bars.loc[0, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, stop_loss_price=1990.0),
        3: _sig(time=bars.loc[3, "time"], symbol="XAUUSD", signal_type=SignalType.LONG, stop_loss_price=1990.0),
    }
    engine = BacktestEngine(config=cfg)
    r1 = engine.run(bars, ScriptedEventStrategy(signals))
    r2 = engine.run(bars, ScriptedEventStrategy(signals))

    pd.testing.assert_frame_equal(r1.equity, r2.equity)
    pd.testing.assert_frame_equal(r1.trades, r2.trades)
    pd.testing.assert_frame_equal(r1.orders, r2.orders)
    pd.testing.assert_frame_equal(r1.ledger, r2.ledger)
    pd.testing.assert_frame_equal(r1.risk_events, r2.risk_events)
