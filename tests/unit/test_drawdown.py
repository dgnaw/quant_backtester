"""
Unit tests for Maximum Drawdown Circuit Breaker & DrawdownMonitor (Group F - Task F4 / DF5).
Covers all 8 required test cases.
"""

from __future__ import annotations

import math
import pandas as pd

from quant_backtester.config import InstrumentConfig, RiskConfig
from quant_backtester.models.order import OrderIntent, OrderSide, OrderType
from quant_backtester.portfolio.position import Position, PositionSide
from quant_backtester.risk import HaltReason, RiskContext, SimpleRiskManager


def _inst() -> InstrumentConfig:
    return InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10)


def test_hwm_tracks_rising_equity():
    """1. High Water Mark (HWM) updates upward as equity reaches new peaks."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_pct=0.10),
        instrument_config=_inst(),
    )
    t1 = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    rm.drawdown_monitor.update(t1, current_equity=10000.0, halt_manager=rm.halt_manager)
    assert math.isclose(rm.drawdown_monitor.high_water_mark_equity, 10000.0)

    rm.drawdown_monitor.update(t1, current_equity=12500.0, halt_manager=rm.halt_manager)
    assert math.isclose(rm.drawdown_monitor.high_water_mark_equity, 12500.0)
    assert math.isclose(rm.drawdown_monitor.current_drawdown_amount, 0.0)


def test_max_drawdown_pct_breach_from_peak_triggers_halt():
    """
    2. Equity rises from 10,000 -> 12,000 (HWM), then drops to 10,600 (drawdown = 1,400 / 12,000 = 11.67% > 10%).
    Triggers HALTED_MAX_DRAWDOWN even though equity (10,600) is still above initial balance (10,000)!
    """
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_pct=0.10),
        instrument_config=_inst(),
    )
    t = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    rm.drawdown_monitor.update(t, current_equity=10000.0, halt_manager=rm.halt_manager)
    rm.drawdown_monitor.update(t, current_equity=12000.0, halt_manager=rm.halt_manager)
    res = rm.drawdown_monitor.update(t, current_equity=10600.0, halt_manager=rm.halt_manager)
    assert not res.passed
    assert res.reject_code == "MAX_DRAWDOWN_LIMIT_REACHED"
    assert rm.halt_manager.has_reason(HaltReason.HALTED_MAX_DRAWDOWN)


def test_max_drawdown_amount_breach_triggers_halt():
    """3. Reaching max_drawdown_amount triggers HALTED_MAX_DRAWDOWN and rejects new entries."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_amount=1000.0),
        instrument_config=_inst(),
    )
    t = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    rm.drawdown_monitor.update(t, current_equity=10000.0, halt_manager=rm.halt_manager)
    res = rm.drawdown_monitor.update(t, current_equity=8950.0, halt_manager=rm.halt_manager)
    assert not res.passed
    assert res.reject_code == "MAX_DRAWDOWN_LIMIT_REACHED"

    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
    dec = rm.evaluate_order_intent(intent, 8950.0, 8950.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "MAX_DRAWDOWN_LIMIT_REACHED"


def test_max_drawdown_halt_does_not_reset_on_new_trading_day():
    """4. Unlike Daily Loss, HALTED_MAX_DRAWDOWN is persistent across subsequent trading days."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_pct=0.10, max_daily_loss_pct=0.05),
        instrument_config=_inst(),
    )
    d1 = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    d2 = pd.Timestamp("2025-01-03 10:00:00", tz="UTC")

    rm.update_portfolio_state(RiskContext(timestamp=d1, current_balance=10000.0, current_equity=10000.0))
    rm.update_portfolio_state(RiskContext(timestamp=d1, current_balance=8800.0, current_equity=8800.0))
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_MAX_DRAWDOWN)

    # On Day 2, HALTED_DAILY_LOSS clears, but HALTED_MAX_DRAWDOWN remains active!
    rm.update_portfolio_state(RiskContext(timestamp=d2, current_balance=8800.0, current_equity=8800.0))
    assert not rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_MAX_DRAWDOWN)
    assert rm.halt_manager.is_halted


def test_withdrawal_adjusts_hwm_without_triggering_false_drawdown():
    """5. Withdrawing $2,000 from $10,000 adjusts HWM to $8,000 so drawdown remains 0."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_amount=500.0),
        instrument_config=_inst(),
    )
    t = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.drawdown_monitor.update(t, current_equity=10000.0, net_deposits=0.0, halt_manager=rm.halt_manager)
    res = rm.drawdown_monitor.update(
        t, current_equity=8000.0, net_deposits=-2000.0, halt_manager=rm.halt_manager
    )
    assert res.passed
    assert math.isclose(rm.drawdown_monitor.high_water_mark_equity, 8000.0)
    assert math.isclose(rm.drawdown_monitor.current_drawdown_amount, 0.0)


def test_deposit_adjusts_hwm_so_trading_drawdown_is_not_masked():
    """6. Depositing $2,000 raises HWM from $10,000 to $12,000; if equity is $10,800, drawdown is $1,200."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_amount=1000.0),
        instrument_config=_inst(),
    )
    t = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.drawdown_monitor.update(t, current_equity=10000.0, net_deposits=0.0, halt_manager=rm.halt_manager)
    res = rm.drawdown_monitor.update(
        t, current_equity=10800.0, net_deposits=2000.0, halt_manager=rm.halt_manager
    )
    assert not res.passed
    assert math.isclose(rm.drawdown_monitor.current_drawdown_amount, 1200.0)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_MAX_DRAWDOWN)


def test_liquidate_on_max_drawdown_flag_sets_liquidation_request():
    """7. When liquidate_on_max_drawdown=True, breaching max drawdown sets liquidation_requested=True once."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_pct=0.10, liquidate_on_max_drawdown=True),
        instrument_config=_inst(),
    )
    t = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.drawdown_monitor.update(t, current_equity=10000.0, halt_manager=rm.halt_manager)
    rm.drawdown_monitor.update(t, current_equity=8900.0, halt_manager=rm.halt_manager)
    assert rm.drawdown_monitor.consume_liquidation_request() is True
    # Subsequent consume returns False (one-time trigger)
    assert rm.drawdown_monitor.consume_liquidation_request() is False


def test_exit_order_permitted_under_max_drawdown_halt():
    """8. Risk-reducing exit orders are permitted even when HALTED_MAX_DRAWDOWN is active."""
    inst = _inst()
    rm = SimpleRiskManager(
        config=RiskConfig(max_drawdown_pct=0.10),
        instrument_config=inst,
    )
    t = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.drawdown_monitor.update(t, current_equity=10000.0, halt_manager=rm.halt_manager)
    rm.drawdown_monitor.update(t, current_equity=8500.0, halt_manager=rm.halt_manager)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_MAX_DRAWDOWN)

    pos = Position(symbol="XAUUSD", instrument_config=inst, position_id="POS-1")
    pos.open(
        side=PositionSide.SHORT,
        price=2000.0,
        units=20.0,
        quantity=0.20,
        timestamp=t,
        order_id="ORD-1",
        fill_id="FILL-1",
    )
    exit_intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.20,
        reason="STRATEGY_EXIT",
    )
    ctx = RiskContext(
        timestamp=t,
        current_balance=8500.0,
        current_equity=8500.0,
        active_positions={"XAUUSD": pos},
    )
    dec = rm.evaluate_order_intent(exit_intent, 8500.0, 8500.0, context=ctx)
    assert dec.is_approved
