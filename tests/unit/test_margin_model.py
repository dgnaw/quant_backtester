"""
Unit tests for LeverageMarginModel, Margin Call & Stop-Out (Group F - Task F6 / DF7).
Covers all 10 required test cases.
"""

from __future__ import annotations

import math
import pandas as pd
import pytest

from quant_backtester.config import InstrumentConfig, RiskConfig
from quant_backtester.models.order import Order, OrderIntent, OrderOrigin, OrderSide, OrderStatus, OrderType
from quant_backtester.portfolio.position import Position, PositionSide
from quant_backtester.risk import (
    HaltReason,
    LeverageMarginModel,
    RiskContext,
    SimpleRiskManager,
)


def _xau_inst(leverage: float | None = None) -> InstrumentConfig:
    return InstrumentConfig(
        symbol="XAUUSD",
        base_currency="XAU",
        quote_currency="USD",
        contract_size=100.0,
        lot_size=0.10,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        leverage=leverage,
    )


def _open_pos(
    symbol: str,
    side: PositionSide,
    qty: float,
    price: float,
    inst: InstrumentConfig,
    mark_price: float | None = None,
) -> Position:
    pos = Position(symbol=symbol, instrument_config=inst, position_id=f"POS-{symbol}")
    pos.open(
        side=side,
        price=price,
        units=qty * inst.contract_size,
        quantity=qty,
        timestamp=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        order_id="ORD-1",
        fill_id="FILL-1",
    )
    if mark_price is not None:
        pos.unrealized_pnl(mark_price)
    return pos


def test_calculate_required_margin_basic():
    """
    1. 1.0 lot XAUUSD (100 oz) @ 2000.0 with leverage 100:1 -> notional $200,000 -> margin $2,000.
    """
    model = LeverageMarginModel(RiskConfig(leverage=100.0))
    req = model.calculate_required_margin(
        units=100.0,
        price=2000.0,
        instrument_config=_xau_inst(),
        account_currency="USD",
    )
    assert math.isclose(req, 2000.0)


def test_instrument_leverage_overrides_account_leverage():
    """2. InstrumentConfig.leverage overrides RiskConfig.leverage."""
    model = LeverageMarginModel(RiskConfig(leverage=100.0))
    req = model.calculate_required_margin(
        units=100.0,
        price=2000.0,
        instrument_config=_xau_inst(leverage=50.0),
        account_currency="USD",
    )
    assert math.isclose(req, 4000.0)


def test_snapshot_when_flat_has_infinite_margin_level():
    """3. When no positions are open, used_margin=0, free_margin=equity, margin_level_pct=inf."""
    model = LeverageMarginModel(RiskConfig(leverage=100.0))
    snap = model.compute_snapshot(equity=10000.0, active_positions={})
    assert math.isclose(snap.used_margin, 0.0)
    assert math.isclose(snap.free_margin, 10000.0)
    assert math.isinf(snap.margin_level_pct)
    assert not snap.is_margin_call
    assert not snap.is_stop_out


def test_snapshot_with_open_positions_and_pending_reservations():
    """
    4. Snapshot computes used_margin from active positions and reserved_margin from pending entry orders.
    """
    cfg = RiskConfig(leverage=100.0, reserve_margin_for_pending_orders=True)
    model = LeverageMarginModel(cfg)
    inst = _xau_inst()
    pos = _open_pos("XAUUSD", PositionSide.LONG, qty=1.0, price=2000.0, inst=inst)
    pending_order = Order(
        order_id="ORD-PEND",
        time=pd.Timestamp("2025-01-01 01:00:00", tz="UTC"),
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.5,
        units=50.0,
        limit_price=2000.0,
        status=OrderStatus.PENDING,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    snap = model.compute_snapshot(
        equity=5000.0,
        active_positions={"XAUUSD": pos},
        pending_entry_orders=[pending_order],
        mark_prices={"XAUUSD": 2000.0, "EURUSD": 2000.0},
        instrument_by_symbol={"XAUUSD": inst, "EURUSD": inst},
        default_instrument=inst,
    )
    assert math.isclose(snap.used_margin, 2000.0)
    assert math.isclose(snap.reserved_margin, 1000.0)
    assert math.isclose(snap.free_margin, 3000.0)
    assert math.isclose(snap.effective_free_margin, 2000.0)
    assert math.isclose(snap.margin_level_pct, 250.0)


def test_pre_trade_margin_rejects_when_insufficient_free_margin():
    """5. Pre-trade margin check rejects with INSUFFICIENT_FREE_MARGIN when required > effective_free_margin."""
    rm = SimpleRiskManager(
        config=RiskConfig(leverage=100.0, enforce_margin_check=True),
        instrument_config=_xau_inst(),
    )
    # 1.0 lot @ 2000 requires $2000 margin; equity is $1500 -> rejected
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )
    dec = rm.evaluate_order_intent(intent, 1500.0, 1500.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "INSUFFICIENT_FREE_MARGIN"


def test_pending_order_on_same_symbol_is_excluded_from_reserved_margin():
    """
    6. DF6/DF7: When submitting a new entry order on symbol S, an existing PENDING entry order
    on symbol S will be superseded by OrderManager, so its reserved margin is not double-counted.
    """
    inst = _xau_inst()
    rm = SimpleRiskManager(
        config=RiskConfig(
            leverage=100.0,
            enforce_margin_check=True,
            reserve_margin_for_pending_orders=True,
        ),
        instrument_config=inst,
    )
    existing_pending = Order(
        order_id="ORD-OLD",
        time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        units=100.0,
        limit_price=2000.0,
        status=OrderStatus.PENDING,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    # Equity = $2500. Existing pending on XAUUSD reserves $2000.
    # New order on XAUUSD for 1.0 lot ($2000 margin) replaces ORD-OLD, so it MUST pass!
    ctx = RiskContext(
        current_balance=2500.0,
        current_equity=2500.0,
        pending_entry_orders=[existing_pending],
        mark_prices={"XAUUSD": 2000.0},
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        limit_price=2000.0,
    )
    dec = rm.evaluate_order_intent(intent, 2500.0, 2500.0, reference_price=2000.0, context=ctx)
    assert dec.is_approved


def test_margin_call_triggers_halt_and_blocks_new_entries():
    """
    7. When margin_level_pct <= margin_call_level_pct (100%), HALTED_MARGIN_CALL is triggered
    and new entry orders are rejected with MARGIN_CALL_ACTIVE.
    """
    inst = _xau_inst()
    rm = SimpleRiskManager(
        config=RiskConfig(
            leverage=100.0,
            enforce_margin_check=True,
            margin_call_level_pct=100.0,
            stop_out_level_pct=50.0,
        ),
        instrument_config=inst,
    )
    pos = _open_pos("XAUUSD", PositionSide.LONG, qty=1.0, price=2000.0, inst=inst, mark_price=1985.0)
    # Used margin @ 1985 = $1985. Equity = $1800 -> margin_level_pct = 90.68% <= 100%
    ctx = RiskContext(
        timestamp=pd.Timestamp("2025-01-01 01:00:00", tz="UTC"),
        current_balance=3300.0,
        current_equity=1800.0,
        active_positions={"XAUUSD": pos},
        mark_prices={"XAUUSD": 1985.0},
    )
    snap = rm.update_portfolio_state(ctx)
    assert snap.is_margin_call
    assert not snap.is_stop_out
    assert rm.halt_manager.has_reason(HaltReason.HALTED_MARGIN_CALL)

    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
    dec = rm.evaluate_order_intent(intent, 3300.0, 1800.0, reference_price=1985.0, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "MARGIN_CALL_ACTIVE"


def test_margin_call_clears_automatically_when_equity_recovers():
    """8. When equity recovers and margin_level_pct > margin_call_level_pct, HALTED_MARGIN_CALL clears."""
    inst = _xau_inst()
    rm = SimpleRiskManager(
        config=RiskConfig(
            leverage=100.0,
            enforce_margin_check=True,
            margin_call_level_pct=100.0,
        ),
        instrument_config=inst,
    )
    pos = _open_pos("XAUUSD", PositionSide.LONG, qty=1.0, price=2000.0, inst=inst, mark_price=1980.0)
    ctx_low = RiskContext(
        current_balance=3500.0,
        current_equity=1800.0,
        active_positions={"XAUUSD": pos},
        mark_prices={"XAUUSD": 1980.0},
    )
    rm.update_portfolio_state(ctx_low)
    assert rm.halt_manager.has_reason(HaltReason.HALTED_MARGIN_CALL)

    # Price recovers to 2010 -> Equity = $3600, used_margin = $2010 -> level = 179.1% > 100%
    ctx_rec = RiskContext(
        current_balance=3500.0,
        current_equity=3600.0,
        active_positions={"XAUUSD": pos},
        mark_prices={"XAUUSD": 2010.0},
    )
    snap_rec = rm.update_portfolio_state(ctx_rec)
    assert not snap_rec.is_margin_call
    assert not rm.halt_manager.has_reason(HaltReason.HALTED_MARGIN_CALL)


def test_stop_out_detection_and_deterministic_priority_sorting():
    """
    9. Stop-Out priority order:
    (1) Largest unrealized loss first (`unrealized_pnl` ascending)
    (2) Largest required margin first (`required_margin` descending)
    (3) Symbol alphabetical (`symbol` A -> Z)
    """
    inst_a = InstrumentConfig(symbol="AUDUSD", contract_size=100.0)
    inst_b = InstrumentConfig(symbol="EURUSD", contract_size=100.0)
    inst_c = InstrumentConfig(symbol="GBPUSD", contract_size=100.0)
    inst_d = InstrumentConfig(symbol="XAUUSD", contract_size=100.0)

    # POS C: unrealized_pnl = -500 (worst loss -> #1)
    pos_c = _open_pos("GBPUSD", PositionSide.LONG, qty=1.0, price=100.0, inst=inst_c, mark_price=95.0)
    # POS D: unrealized_pnl = -200, margin @ 2000 = 2000 (tie on loss with A & B, but larger margin -> #2)
    pos_d = _open_pos("XAUUSD", PositionSide.LONG, qty=1.0, price=2002.0, inst=inst_d, mark_price=2000.0)
    # POS A & B: both have unrealized_pnl = -200 and same margin @ 100 = 100 -> tie-breaker A->Z: AUDUSD (#3), EURUSD (#4)
    pos_b = _open_pos("EURUSD", PositionSide.LONG, qty=1.0, price=102.0, inst=inst_b, mark_price=100.0)
    pos_a = _open_pos("AUDUSD", PositionSide.LONG, qty=1.0, price=102.0, inst=inst_a, mark_price=100.0)

    model = LeverageMarginModel(RiskConfig(leverage=100.0, stop_out_level_pct=50.0))
    ordered = model.select_positions_for_stop_out(
        active_positions={"EURUSD": pos_b, "XAUUSD": pos_d, "GBPUSD": pos_c, "AUDUSD": pos_a},
        mark_prices={"GBPUSD": 95.0, "XAUUSD": 2000.0, "EURUSD": 100.0, "AUDUSD": 100.0},
        instrument_by_symbol={"AUDUSD": inst_a, "EURUSD": inst_b, "GBPUSD": inst_c, "XAUUSD": inst_d},
    )
    assert [p.symbol for p in ordered] == ["GBPUSD", "XAUUSD", "AUDUSD", "EURUSD"]


def test_invalid_margin_call_and_stop_out_config_raises_error():
    """10. stop_out_level_pct > margin_call_level_pct raises ValueError."""
    with pytest.raises(ValueError, match="stop_out_level_pct"):
        RiskConfig(margin_call_level_pct=50.0, stop_out_level_pct=80.0)
