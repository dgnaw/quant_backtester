"""
Unit tests for Exposure Limits & Correlation Risk Control (Group F - Tasks F5 & F7 / DF6 & DF8).
Covers all 10 required test cases.
"""

from __future__ import annotations

import pandas as pd

from quant_backtester.config import InstrumentConfig, MissingCorrelationPolicy, RiskConfig
from quant_backtester.models.order import Order, OrderIntent, OrderOrigin, OrderSide, OrderStatus, OrderType
from quant_backtester.portfolio.position import Position, PositionSide
from quant_backtester.risk import RiskContext, SimpleRiskManager


def _inst(symbol: str = "XAUUSD", contract_size: float = 100.0) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol,
        base_currency=symbol[:3],
        quote_currency="USD",
        contract_size=contract_size,
        lot_size=0.10,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
    )


def _open_pos(symbol: str, side: PositionSide, qty: float, price: float, contract_size: float = 100.0) -> Position:
    inst = _inst(symbol, contract_size)
    pos = Position(symbol=symbol, instrument_config=inst, position_id=f"POS-{symbol}")
    pos.open(
        side=side,
        price=price,
        units=qty * contract_size,
        quantity=qty,
        timestamp=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        order_id=f"ORD-{symbol}",
        fill_id=f"FILL-{symbol}",
    )
    return pos


def test_symbol_lot_limit_rejects_excess_scale_in():
    """1. Exceeding max_symbol_lots is rejected with SYMBOL_LOT_LIMIT_EXCEEDED."""
    inst = _inst("XAUUSD")
    rm = SimpleRiskManager(
        config=RiskConfig(max_symbol_lots=1.0),
        instrument_config=inst,
    )
    pos = _open_pos("XAUUSD", PositionSide.LONG, qty=0.70, price=2000.0)
    ctx = RiskContext(
        current_balance=10000.0,
        current_equity=10000.0,
        active_positions={"XAUUSD": pos},
        mark_prices={"XAUUSD": 2000.0},
    )
    # Adding 0.40 lot same side -> 1.10 > 1.0 -> rejected
    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.40)
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "SYMBOL_LOT_LIMIT_EXCEEDED"


def test_symbol_notional_limit_rejects_excess_notional():
    """2. Exceeding max_symbol_notional is rejected with SYMBOL_NOTIONAL_LIMIT_EXCEEDED."""
    inst = _inst("XAUUSD", contract_size=100.0)
    rm = SimpleRiskManager(
        config=RiskConfig(max_symbol_notional=150_000.0),
        instrument_config=inst,
    )
    # 1.0 lot @ 2000 = $200,000 notional > $150,000
    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "SYMBOL_NOTIONAL_LIMIT_EXCEEDED"


def test_account_lot_and_notional_limits_across_multiple_symbols():
    """3. Account lot and notional limits sum gross exposure across all open symbols."""
    inst_xau = _inst("XAUUSD", contract_size=100.0)
    inst_eur = _inst("EURUSD", contract_size=100_000.0)
    rm = SimpleRiskManager(
        config=RiskConfig(max_account_lots=1.5, max_account_notional=250_000.0),
        instrument_config=inst_xau,
        instrument_by_symbol={"XAUUSD": inst_xau, "EURUSD": inst_eur},
    )
    pos_eur = _open_pos("EURUSD", PositionSide.LONG, qty=1.0, price=1.10, contract_size=100_000.0)
    ctx = RiskContext(
        current_balance=50000.0,
        current_equity=50000.0,
        active_positions={"EURUSD": pos_eur},
        mark_prices={"EURUSD": 1.10, "XAUUSD": 2000.0},
        instrument_by_symbol={"XAUUSD": inst_xau, "EURUSD": inst_eur},
    )
    # 1) Lot limit check: EURUSD has 1.0 lot; requesting 0.60 lot XAUUSD -> 1.60 > 1.50
    intent_lots = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.60)
    dec1 = rm.evaluate_order_intent(intent_lots, 50000.0, 50000.0, reference_price=2000.0, context=ctx)
    assert not dec1.is_approved
    assert dec1.reject_reason == "ACCOUNT_LOT_LIMIT_EXCEEDED"

    # 2) Notional limit check: EURUSD has $110,000; requesting 0.80 lot XAUUSD ($160,000) with higher lot cap
    rm.config.max_account_lots = 5.0
    intent_notional = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.80)
    dec2 = rm.evaluate_order_intent(intent_notional, 50000.0, 50000.0, reference_price=2000.0, context=ctx)
    assert not dec2.is_approved
    assert dec2.reject_reason == "ACCOUNT_NOTIONAL_LIMIT_EXCEEDED"


def test_pending_orders_on_other_symbols_included_but_same_symbol_superseded():
    """
    4. DF6: Pending entry orders on k != S are included in account exposure,
    while an existing pending entry order on target symbol S is replaced (not double-counted).
    """
    inst_xau = _inst("XAUUSD", contract_size=100.0)
    inst_eur = _inst("EURUSD", contract_size=100.0)
    rm = SimpleRiskManager(
        config=RiskConfig(
            max_symbol_lots=1.0,
            max_account_lots=1.5,
            include_pending_orders_in_exposure=True,
        ),
        instrument_config=inst_xau,
        instrument_by_symbol={"XAUUSD": inst_xau, "EURUSD": inst_eur},
    )
    pend_xau = Order(
        order_id="ORD-XAU-1",
        time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.80,
        units=80.0,
        limit_price=1990.0,
        status=OrderStatus.PENDING,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    pend_eur = Order(
        order_id="ORD-EUR-1",
        time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.60,
        units=60.0,
        limit_price=1.10,
        status=OrderStatus.PENDING,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    ctx = RiskContext(
        current_balance=10000.0,
        current_equity=10000.0,
        pending_entry_orders=[pend_xau, pend_eur],
    )
    # Submitting 0.80 lot on XAUUSD supersedes pend_xau (0.80) -> symbol XAUUSD is 0.80 <= 1.0,
    # and account total is 0.80 (XAUUSD) + 0.60 (EURUSD pending) = 1.40 <= 1.5 -> APPROVED!
    intent_ok = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.80, limit_price=1995.0)
    dec_ok = rm.evaluate_order_intent(intent_ok, 10000.0, 10000.0, reference_price=1995.0, context=ctx)
    assert dec_ok.is_approved

    # Submitting 1.00 lot on XAUUSD -> 1.00 (XAUUSD) + 0.60 (EURUSD pending) = 1.60 > 1.5 -> REJECTED!
    intent_big = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=1.00, limit_price=1995.0)
    dec_big = rm.evaluate_order_intent(intent_big, 10000.0, 10000.0, reference_price=1995.0, context=ctx)
    assert not dec_big.is_approved
    assert dec_big.reject_reason == "ACCOUNT_LOT_LIMIT_EXCEEDED"


def test_correlation_positive_same_direction_exceeds_max_correlated_positions():
    """
    5. EURUSD LONG + GBPUSD LONG with rho=+0.85 >= 0.70 -> effective_corr = +0.85.
    When max_correlated_positions=1, the second position is rejected with CORRELATED_POSITIONS_LIMIT_EXCEEDED.
    """
    inst_eur = _inst("EURUSD", contract_size=100_000.0)
    inst_gbp = _inst("GBPUSD", contract_size=100_000.0)
    rm = SimpleRiskManager(
        config=RiskConfig(
            enable_correlation_check=True,
            correlation_threshold=0.70,
            max_correlated_positions=1,
            correlation_matrix={"EURUSD": {"GBPUSD": 0.85}},
        ),
        instrument_config=inst_gbp,
        instrument_by_symbol={"EURUSD": inst_eur, "GBPUSD": inst_gbp},
    )
    pos_eur = _open_pos("EURUSD", PositionSide.LONG, qty=0.10, price=1.10, contract_size=100_000.0)
    ctx = RiskContext(
        current_balance=10000.0,
        current_equity=10000.0,
        active_positions={"EURUSD": pos_eur},
        mark_prices={"EURUSD": 1.10, "GBPUSD": 1.25},
        instrument_by_symbol={"EURUSD": inst_eur, "GBPUSD": inst_gbp},
    )
    intent = OrderIntent(symbol="GBPUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.10)
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=1.25, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "CORRELATED_POSITIONS_LIMIT_EXCEEDED"


def test_correlation_positive_opposite_direction_acts_as_natural_hedge_and_passes():
    """
    6. EURUSD LONG + GBPUSD SHORT with rho=+0.85 -> direction_sign = -1 -> effective_corr = -0.85 < 0.70.
    Approved because opposite direction on positively correlated pair is not concentrated directional risk.
    """
    inst_eur = _inst("EURUSD", contract_size=100_000.0)
    inst_gbp = _inst("GBPUSD", contract_size=100_000.0)
    rm = SimpleRiskManager(
        config=RiskConfig(
            enable_correlation_check=True,
            correlation_threshold=0.70,
            max_correlated_positions=1,
            correlation_matrix={"EURUSD": {"GBPUSD": 0.85}},
        ),
        instrument_config=inst_gbp,
        instrument_by_symbol={"EURUSD": inst_eur, "GBPUSD": inst_gbp},
    )
    pos_eur = _open_pos("EURUSD", PositionSide.LONG, qty=0.10, price=1.10, contract_size=100_000.0)
    ctx = RiskContext(
        current_balance=10000.0,
        current_equity=10000.0,
        active_positions={"EURUSD": pos_eur},
        mark_prices={"EURUSD": 1.10, "GBPUSD": 1.25},
        instrument_by_symbol={"EURUSD": inst_eur, "GBPUSD": inst_gbp},
    )
    intent = OrderIntent(symbol="GBPUSD", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.10)
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=1.25, context=ctx)
    assert dec.is_approved


def test_negative_correlation_opposite_direction_is_concentrated_risk():
    """
    7. EURUSD LONG + USDCHF SHORT with rho=-0.90 -> direction_sign = -1 -> effective_corr = (-0.90)*(-1) = +0.90 >= 0.70!
    Both positions bet against USD, so effective correlation is +0.90 -> rejected when max_correlated_positions=1.
    """
    inst_eur = _inst("EURUSD", contract_size=100_000.0)
    inst_chf = _inst("USDCHF", contract_size=100_000.0)
    rm = SimpleRiskManager(
        config=RiskConfig(
            enable_correlation_check=True,
            correlation_threshold=0.70,
            max_correlated_positions=1,
            correlation_matrix={"EURUSD": {"USDCHF": -0.90}},
        ),
        instrument_config=inst_chf,
        instrument_by_symbol={"EURUSD": inst_eur, "USDCHF": inst_chf},
    )
    pos_eur = _open_pos("EURUSD", PositionSide.LONG, qty=0.10, price=1.10, contract_size=100_000.0)
    ctx = RiskContext(
        current_balance=10000.0,
        current_equity=10000.0,
        active_positions={"EURUSD": pos_eur},
        mark_prices={"EURUSD": 1.10, "USDCHF": 0.90},
        instrument_by_symbol={"EURUSD": inst_eur, "USDCHF": inst_chf},
    )
    intent = OrderIntent(symbol="USDCHF", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.10)
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=0.90, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "CORRELATED_POSITIONS_LIMIT_EXCEEDED"


def test_correlated_notional_limit_rejects_excess_cluster_notional():
    """8. Exceeding max_correlated_notional rejects with CORRELATED_EXPOSURE_LIMIT_EXCEEDED."""
    inst_eur = _inst("EURUSD", contract_size=100_000.0)
    inst_gbp = _inst("GBPUSD", contract_size=100_000.0)
    rm = SimpleRiskManager(
        config=RiskConfig(
            enable_correlation_check=True,
            correlation_threshold=0.70,
            max_correlated_positions=3,
            max_correlated_notional=200_000.0,
            correlation_matrix={"EURUSD": {"GBPUSD": 0.85}},
        ),
        instrument_config=inst_gbp,
        instrument_by_symbol={"EURUSD": inst_eur, "GBPUSD": inst_gbp},
    )
    # EURUSD LONG 1.0 lot @ 1.10 = $110,000
    pos_eur = _open_pos("EURUSD", PositionSide.LONG, qty=1.0, price=1.10, contract_size=100_000.0)
    ctx = RiskContext(
        current_balance=50000.0,
        current_equity=50000.0,
        active_positions={"EURUSD": pos_eur},
        mark_prices={"EURUSD": 1.10, "GBPUSD": 1.25},
        instrument_by_symbol={"EURUSD": inst_eur, "GBPUSD": inst_gbp},
    )
    # GBPUSD BUY 1.0 lot @ 1.25 = $125,000 -> total correlated notional = $235,000 > $200,000
    intent = OrderIntent(symbol="GBPUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
    dec = rm.evaluate_order_intent(intent, 50000.0, 50000.0, reference_price=1.25, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "CORRELATED_EXPOSURE_LIMIT_EXCEEDED"


def test_missing_correlation_policy_reject():
    """9. MissingCorrelationPolicy.REJECT rejects with MISSING_CORRELATION_DATA when pair not in matrix."""
    inst_eur = _inst("EURUSD")
    inst_aud = _inst("AUDUSD")
    rm = SimpleRiskManager(
        config=RiskConfig(
            enable_correlation_check=True,
            max_correlated_positions=2,
            missing_correlation_policy=MissingCorrelationPolicy.REJECT,
            correlation_matrix={},
        ),
        instrument_config=inst_aud,
    )
    pos_eur = _open_pos("EURUSD", PositionSide.LONG, qty=0.10, price=1.10)
    ctx = RiskContext(
        current_balance=10000.0,
        current_equity=10000.0,
        active_positions={"EURUSD": pos_eur},
    )
    intent = OrderIntent(symbol="AUDUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.10)
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=0.65, context=ctx)
    assert not dec.is_approved
    assert dec.reject_reason == "MISSING_CORRELATION_DATA"


def test_missing_correlation_policy_ignore_and_conservative_one():
    """10. MissingCorrelationPolicy.IGNORE ignores unknown pairs; CONSERVATIVE_ONE treats them as +1.0."""
    inst_eur = _inst("EURUSD")
    inst_aud = _inst("AUDUSD")
    pos_eur = _open_pos("EURUSD", PositionSide.LONG, qty=0.10, price=1.10)
    ctx = RiskContext(
        current_balance=10000.0,
        current_equity=10000.0,
        active_positions={"EURUSD": pos_eur},
    )
    intent = OrderIntent(symbol="AUDUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.10)

    # IGNORE -> approved even with max_correlated_positions=1
    rm_ignore = SimpleRiskManager(
        config=RiskConfig(
            enable_correlation_check=True,
            max_correlated_positions=1,
            missing_correlation_policy=MissingCorrelationPolicy.IGNORE,
            correlation_matrix={},
        ),
        instrument_config=inst_aud,
    )
    assert rm_ignore.evaluate_order_intent(intent, 10000.0, 10000.0, 0.65, context=ctx).is_approved

    # CONSERVATIVE_ONE -> treated as effective_corr=1.0 -> rejected when max_correlated_positions=1
    rm_cons = SimpleRiskManager(
        config=RiskConfig(
            enable_correlation_check=True,
            max_correlated_positions=1,
            missing_correlation_policy=MissingCorrelationPolicy.CONSERVATIVE_ONE,
            correlation_matrix={},
        ),
        instrument_config=inst_aud,
    )
    dec_cons = rm_cons.evaluate_order_intent(intent, 10000.0, 10000.0, 0.65, context=ctx)
    assert not dec_cons.is_approved
    assert dec_cons.reject_reason == "CORRELATED_POSITIONS_LIMIT_EXCEEDED"
