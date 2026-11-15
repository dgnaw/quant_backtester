"""
Unit tests for Daily Loss Limit & DailyLossMonitor (Group F - Task F3 / DF4).
Covers all 10 required test cases.
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


def test_day_start_equity_initialized_on_first_bar():
    """1. Day-start equity is initialized on the first bar of the session."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=500.0),
        instrument_config=_inst(),
    )
    ts = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    res = rm.daily_loss_monitor.update(ts, current_equity=10000.0, halt_manager=rm.halt_manager)
    assert res.passed
    assert math.isclose(rm.daily_loss_monitor.day_start_equity, 10000.0)
    assert math.isclose(rm.daily_loss_monitor.current_daily_loss, 0.0)


def test_daily_loss_amount_breach_triggers_halt():
    """2. Reaching max_daily_loss_amount triggers HALTED_DAILY_LOSS and rejects new entries."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=300.0),
        instrument_config=_inst(),
    )
    ts1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    ts2 = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.daily_loss_monitor.update(ts1, current_equity=10000.0, halt_manager=rm.halt_manager)
    res2 = rm.daily_loss_monitor.update(ts2, current_equity=9700.0, halt_manager=rm.halt_manager)
    assert not res2.passed
    assert res2.reject_code == "DAILY_LOSS_LIMIT_REACHED"
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)

    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
    ctx = RiskContext(timestamp=ts2, current_balance=9700.0, current_equity=9700.0)
    dec = rm.evaluate_order_intent(intent, 9700.0, 9700.0, reference_price=2000.0, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "DAILY_LOSS_LIMIT_REACHED"


def test_daily_loss_pct_breach_triggers_halt():
    """3. Reaching max_daily_loss_pct triggers HALTED_DAILY_LOSS."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_pct=0.02),  # 2% of 10,000 = $200
        instrument_config=_inst(),
    )
    ts1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    ts2 = pd.Timestamp("2025-01-02 11:00:00", tz="UTC")
    rm.daily_loss_monitor.update(ts1, current_equity=10000.0, halt_manager=rm.halt_manager)
    res2 = rm.daily_loss_monitor.update(ts2, current_equity=9790.0, halt_manager=rm.halt_manager)
    assert not res2.passed
    assert res2.reject_code == "DAILY_LOSS_LIMIT_REACHED"
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)


def test_daily_loss_includes_unrealized_pnl_drop():
    """4. Floating unrealized loss on open position triggers daily loss limit even before trade closes."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=250.0),
        instrument_config=_inst(),
    )
    ts1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    ts2 = pd.Timestamp("2025-01-02 12:00:00", tz="UTC")
    # Start day with balance=10,000, equity=10,000
    rm.update_portfolio_state(
        RiskContext(timestamp=ts1, current_balance=10000.0, current_equity=10000.0)
    )
    # Balance is still 10,000, but floating loss brings equity to 9,700 (-$300 floating)
    rm.update_portfolio_state(
        RiskContext(timestamp=ts2, current_balance=10000.0, current_equity=9700.0)
    )
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)


def test_daily_loss_halt_persists_within_same_day_even_if_equity_bounces():
    """5. Once HALTED_DAILY_LOSS is triggered on day D, it stays halted for the rest of day D."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=300.0),
        instrument_config=_inst(),
    )
    ts1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    ts2 = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    ts3 = pd.Timestamp("2025-01-02 14:00:00", tz="UTC")
    rm.daily_loss_monitor.update(ts1, current_equity=10000.0, halt_manager=rm.halt_manager)
    rm.daily_loss_monitor.update(ts2, current_equity=9650.0, halt_manager=rm.halt_manager)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)

    # Later on the same day, floating PnL bounces back to 9900
    rm.daily_loss_monitor.update(ts3, current_equity=9900.0, halt_manager=rm.halt_manager)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)


def test_daily_loss_resets_automatically_on_next_trading_day():
    """6. Transitioning to the next trading day resets day_start_equity and clears HALTED_DAILY_LOSS."""
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=300.0),
        instrument_config=_inst(),
    )
    day1_t1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    day1_t2 = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
    day2_t1 = pd.Timestamp("2025-01-03 09:00:00", tz="UTC")

    rm.daily_loss_monitor.update(day1_t1, current_equity=10000.0, halt_manager=rm.halt_manager)
    rm.daily_loss_monitor.update(day1_t2, current_equity=9600.0, halt_manager=rm.halt_manager)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)

    # New day!
    res_d2 = rm.daily_loss_monitor.update(day2_t1, current_equity=9600.0, halt_manager=rm.halt_manager)
    assert res_d2.passed
    assert not rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)
    assert math.isclose(rm.daily_loss_monitor.day_start_equity, 9600.0)

    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
    ctx_d2 = RiskContext(timestamp=day2_t1, current_balance=9600.0, current_equity=9600.0)
    dec = rm.evaluate_order_intent(intent, 9600.0, 9600.0, reference_price=2000.0, context=ctx_d2)
    assert dec.is_approved


def test_daily_loss_respects_custom_trading_timezone():
    """
    7. With trading_timezone='America/New_York' (UTC-5 in winter):
    2025-01-02 23:00 UTC is 18:00 NY (Jan 2), and 2025-01-03 01:00 UTC is 20:00 NY (still Jan 2!).
    New day in NY starts at 2025-01-03 05:00 UTC (00:00 NY Jan 3).
    """
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=200.0, trading_timezone="America/New_York"),
        instrument_config=_inst(),
    )
    t_jan2_18ny = pd.Timestamp("2025-01-02 23:00:00", tz="UTC")
    t_jan2_20ny = pd.Timestamp("2025-01-03 01:00:00", tz="UTC")
    t_jan3_00ny = pd.Timestamp("2025-01-03 05:00:00", tz="UTC")

    rm.daily_loss_monitor.update(t_jan2_18ny, current_equity=10000.0, halt_manager=rm.halt_manager)
    rm.daily_loss_monitor.update(t_jan2_20ny, current_equity=9750.0, halt_manager=rm.halt_manager)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)

    # At 05:00 UTC (midnight NY), session rolls over to 2025-01-03
    rm.daily_loss_monitor.update(t_jan3_00ny, current_equity=9750.0, halt_manager=rm.halt_manager)
    assert not rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)


def test_intraday_withdrawal_does_not_trigger_false_daily_loss():
    """
    8. Intraday withdrawal of $1,000 reduces equity from $10,000 to $9,000 and net_deposits by -$1,000.
    Adjusted day_start_equity becomes $9,000 -> daily_loss = $0 (no false halt).
    """
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=300.0),
        instrument_config=_inst(),
    )
    t1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    t2 = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.daily_loss_monitor.update(t1, current_equity=10000.0, net_deposits=0.0, halt_manager=rm.halt_manager)
    res = rm.daily_loss_monitor.update(
        t2, current_equity=9000.0, net_deposits=-1000.0, halt_manager=rm.halt_manager
    )
    assert res.passed
    assert math.isclose(rm.daily_loss_monitor.current_daily_loss, 0.0)
    assert not rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)


def test_intraday_deposit_does_not_mask_true_trading_loss():
    """
    9. Intraday deposit of $1,000 raises net_deposits to +$1,000.
    If trading lost $400 (so equity is $10,600), adjusted_day_start_equity = $11,000 -> daily_loss = $400 >= $300 -> HALT!
    """
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=300.0),
        instrument_config=_inst(),
    )
    t1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    t2 = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.daily_loss_monitor.update(t1, current_equity=10000.0, net_deposits=0.0, halt_manager=rm.halt_manager)
    res = rm.daily_loss_monitor.update(
        t2, current_equity=10600.0, net_deposits=1000.0, halt_manager=rm.halt_manager
    )
    assert not res.passed
    assert math.isclose(rm.daily_loss_monitor.current_daily_loss, 400.0)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)


def test_exit_order_allowed_when_halted_by_daily_loss():
    """10. Risk-reducing exit orders are still approved when HALTED_DAILY_LOSS is active."""
    inst = _inst()
    rm = SimpleRiskManager(
        config=RiskConfig(max_daily_loss_amount=300.0),
        instrument_config=inst,
    )
    t1 = pd.Timestamp("2025-01-02 09:00:00", tz="UTC")
    t2 = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    rm.daily_loss_monitor.update(t1, current_equity=10000.0, halt_manager=rm.halt_manager)
    rm.daily_loss_monitor.update(t2, current_equity=9600.0, halt_manager=rm.halt_manager)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS)

    pos = Position(symbol="XAUUSD", instrument_config=inst, position_id="POS-1")
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        units=10.0,
        quantity=0.10,
        timestamp=t1,
        order_id="ORD-1",
        fill_id="FILL-1",
    )
    exit_intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.10,
        reason="STRATEGY_EXIT",
    )
    ctx = RiskContext(
        timestamp=t2,
        current_balance=9600.0,
        current_equity=9600.0,
        active_positions={"XAUUSD": pos},
    )
    dec = rm.evaluate_order_intent(exit_intent, 9600.0, 9600.0, context=ctx)
    assert dec.is_approved
