"""
Unit tests for TradingHaltManager, Exit Order Bypass Validation, RiskDecision & RiskAuditLog (Group F - Task F8 / DF9 & DF10).
Covers all 10 required test cases.
"""

from __future__ import annotations

import pandas as pd

from quant_backtester.config import InstrumentConfig, RiskConfig
from quant_backtester.models.order import OrderIntent, OrderSide, OrderType
from quant_backtester.portfolio.position import Position, PositionSide
from quant_backtester.risk import (
    HaltReason,
    RISK_AUDIT_COLUMNS,
    RiskContext,
    RiskDecisionStatus,
    SimpleRiskManager,
)


def _inst() -> InstrumentConfig:
    return InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10)


def _open_pos(side: PositionSide = PositionSide.LONG, qty: float = 0.50) -> Position:
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst, position_id="POS-1")
    pos.open(
        side=side,
        price=2000.0,
        units=qty * inst.contract_size,
        quantity=qty,
        timestamp=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        order_id="ORD-1",
        fill_id="FILL-1",
    )
    return pos


def test_manual_halt_blocks_new_entry_orders():
    """1. Triggering HALTED_MANUAL blocks new entry orders with TRADING_HALTED_MANUAL."""
    rm = SimpleRiskManager(instrument_config=_inst())
    rm.halt_manager.trigger_halt(HaltReason.HALTED_MANUAL, details="User emergency stop")
    assert rm.halt_manager.is_halted

    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.10)
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert dec.status == RiskDecisionStatus.REJECTED
    assert dec.reject_reason == "TRADING_HALTED_MANUAL"


def test_multiple_simultaneous_halt_reasons_require_all_cleared():
    """2. Multiple active HaltReasons keep system halted until every reason is cleared."""
    rm = SimpleRiskManager(instrument_config=_inst())
    rm.halt_manager.trigger_halt(HaltReason.HALTED_DAILY_LOSS)
    rm.halt_manager.trigger_halt(HaltReason.HALTED_MAX_DRAWDOWN)
    assert len(rm.halt_manager.active_reasons) == 2

    # Clearing daily loss leaves max drawdown active
    rm.halt_manager.clear_halt(HaltReason.HALTED_DAILY_LOSS)
    assert rm.halt_manager.is_halted
    assert rm.halt_manager.primary_reject_code == "MAX_DRAWDOWN_LIMIT_REACHED"

    # Clearing max drawdown restores normal state
    rm.halt_manager.clear_halt(HaltReason.HALTED_MAX_DRAWDOWN)
    assert not rm.halt_manager.is_halted


def test_full_exit_order_bypasses_active_halt():
    """3. Full exit order (opposite side, qty == pos.quantity) is approved while halted."""
    rm = SimpleRiskManager(instrument_config=_inst())
    rm.halt_manager.trigger_halt(HaltReason.HALTED_MAX_DRAWDOWN)
    pos = _open_pos(PositionSide.LONG, qty=0.50)

    exit_intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.50,
        reason="STRATEGY_EXIT",
    )
    ctx = RiskContext(active_positions={"XAUUSD": pos}, current_balance=9000.0, current_equity=9000.0)
    dec = rm.evaluate_order_intent(exit_intent, 9000.0, 9000.0, context=ctx)
    assert dec.is_approved
    assert dec.sized_quantity == 0.50


def test_partial_close_exit_order_bypasses_active_halt():
    """4. Partial close exit order (opposite side, qty < pos.quantity) is approved while halted."""
    rm = SimpleRiskManager(instrument_config=_inst())
    rm.halt_manager.trigger_halt(HaltReason.HALTED_DAILY_LOSS)
    pos = _open_pos(PositionSide.SHORT, qty=0.50)

    partial_exit = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.20,
        reason="STRATEGY_EXIT",
    )
    ctx = RiskContext(active_positions={"XAUUSD": pos}, current_balance=9000.0, current_equity=9000.0)
    dec = rm.evaluate_order_intent(partial_exit, 9000.0, 9000.0, context=ctx)
    assert dec.is_approved
    assert dec.sized_quantity == 0.20


def test_exit_order_rejected_when_no_open_position():
    """5. Exit order when position is FLAT is rejected with NO_OPEN_POSITION_TO_EXIT."""
    rm = SimpleRiskManager(instrument_config=_inst())
    exit_intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.10,
        reason="STRATEGY_EXIT",
    )
    ctx = RiskContext(active_positions={}, current_balance=10000.0, current_equity=10000.0)
    dec = rm.evaluate_order_intent(exit_intent, 10000.0, 10000.0, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "NO_OPEN_POSITION_TO_EXIT"


def test_exit_order_rejected_when_same_side_as_open_position():
    """6. Exit order with same side as open position is rejected with EXIT_ORDER_SAME_SIDE_AS_POSITION."""
    rm = SimpleRiskManager(instrument_config=_inst())
    pos = _open_pos(PositionSide.LONG, qty=0.50)
    bad_exit = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,  # Same side as LONG!
        order_type=OrderType.MARKET,
        quantity=0.20,
        reason="STRATEGY_EXIT",
    )
    ctx = RiskContext(active_positions={"XAUUSD": pos}, current_balance=10000.0, current_equity=10000.0)
    dec = rm.evaluate_order_intent(bad_exit, 10000.0, 10000.0, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "EXIT_ORDER_SAME_SIDE_AS_POSITION"


def test_exit_order_rejected_when_quantity_exceeds_open_position():
    """
    7. DF9: Exit order with quantity > open_position.quantity would cause a reverse into a new position
    and is rejected with EXIT_QUANTITY_EXCEEDS_POSITION.
    """
    rm = SimpleRiskManager(instrument_config=_inst())
    pos = _open_pos(PositionSide.LONG, qty=0.50)
    over_exit = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.80,  # > 0.50 open position
        reason="STRATEGY_EXIT",
    )
    ctx = RiskContext(active_positions={"XAUUSD": pos}, current_balance=10000.0, current_equity=10000.0)
    dec = rm.evaluate_order_intent(over_exit, 10000.0, 10000.0, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "EXIT_QUANTITY_EXCEEDS_POSITION"


def test_scale_in_and_reverse_entry_blocked_during_halt():
    """8. Scale-in and Reverse Entry orders (origin=STRATEGY_ENTRY) are blocked during Trading Halt."""
    rm = SimpleRiskManager(instrument_config=_inst())
    rm.halt_manager.trigger_halt(HaltReason.HALTED_DAILY_LOSS)
    pos = _open_pos(PositionSide.LONG, qty=0.50)
    ctx = RiskContext(active_positions={"XAUUSD": pos}, current_balance=9000.0, current_equity=9000.0)

    # Scale-in entry
    scale_in = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.10)
    dec_scale = rm.evaluate_order_intent(scale_in, 9000.0, 9000.0, reference_price=2000.0, context=ctx)
    assert not dec_scale.is_approved
    assert dec_scale.reject_reason == "DAILY_LOSS_LIMIT_REACHED"

    # Over-close reverse entry
    reverse_entry = OrderIntent(symbol="XAUUSD", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.00)
    dec_rev = rm.evaluate_order_intent(reverse_entry, 9000.0, 9000.0, reference_price=2000.0, context=ctx)
    assert not dec_rev.is_approved
    assert dec_rev.reject_reason == "DAILY_LOSS_LIMIT_REACHED"


def test_risk_audit_log_records_decisions_and_halt_transitions():
    """9. RiskAuditLog records HALT_TRIGGERED, ORDER_REJECTED, HALT_CLEARED, and ORDER_APPROVED."""
    rm = SimpleRiskManager(instrument_config=_inst())
    t1 = pd.Timestamp("2025-01-01 09:00:00", tz="UTC")
    t2 = pd.Timestamp("2025-01-01 10:00:00", tz="UTC")

    rm.halt_manager.trigger_halt(HaltReason.HALTED_MANUAL, timestamp=t1, details="Test halt")
    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.10)
    rm.evaluate_order_intent(intent, 10000.0, 10000.0, 2000.0, context=RiskContext(timestamp=t1))

    rm.halt_manager.clear_halt(HaltReason.HALTED_MANUAL, timestamp=t2, details="Resume")
    rm.evaluate_order_intent(intent, 10000.0, 10000.0, 2000.0, context=RiskContext(timestamp=t2))

    df = rm.audit_log.to_dataframe()
    assert list(df.columns) == RISK_AUDIT_COLUMNS
    assert list(df["event_type"]) == [
        "HALT_TRIGGERED",
        "ORDER_REJECTED",
        "HALT_CLEARED",
        "ORDER_APPROVED",
    ]


def test_finalize_entry_sl_tp_rejects_pending_order_when_halted():
    """10. If a pending entry order is about to fill at Open/Range after a halt was triggered, finalize_entry_sl_tp rejects it."""
    from quant_backtester.models.order import Order

    rm = SimpleRiskManager(instrument_config=_inst())
    rm.halt_manager.trigger_halt(HaltReason.HALTED_MAX_DRAWDOWN)

    order = Order(
        order_id="ORD-PEND",
        time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.10,
        units=10.0,
        limit_price=1995.0,
    )
    ok, reason, _, _ = rm.finalize_entry_sl_tp(order, proposed_fill_price=1995.0)
    assert not ok
    assert reason == "MAX_DRAWDOWN_LIMIT_REACHED"
