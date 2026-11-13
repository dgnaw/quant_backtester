import pandas as pd
import pytest

from quant_backtester.execution.order_manager import OrderManager
from quant_backtester.models.order import OrderOrigin, OrderSide, OrderStatus, OrderType


def test_single_pending_entry_per_symbol_and_superseded_replacement():
    om = OrderManager()
    ts0 = pd.Timestamp("2026-01-01 00:00")
    ts1 = pd.Timestamp("2026-01-01 00:15")

    ord1 = om.create_order(
        time=ts0,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1990.0,
        bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    assert ord1.status == OrderStatus.PENDING
    assert om.get_pending_entry_order("XAUUSD") is ord1

    # New entry order on different symbol (EURUSD) does NOT affect XAUUSD
    ord_eur = om.create_order(
        time=ts0,
        symbol="EURUSD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1000.0,
        limit_price=1.1000,
        bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    assert ord1.status == OrderStatus.PENDING
    assert ord_eur.status == OrderStatus.PENDING

    # Rejected new entry order on XAUUSD must NOT supersede existing pending order ord1
    ord_rejected = om.create_rejected_order(
        time=ts1,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=999.0,
        units=99900.0,
        limit_price=1992.0,
        bar_index=1,
        reject_reason="INVALID_LOT_SIZE",
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    assert ord_rejected.status == OrderStatus.REJECTED
    assert ord1.status == OrderStatus.PENDING
    assert om.get_pending_entry_order("XAUUSD") is ord1

    # Valid new entry order on XAUUSD replaces ord1 -> ord1 becomes CANCELLED with reason SUPERSEDED
    ord2 = om.create_order(
        time=ts1,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.02,
        units=2.0,
        limit_price=1995.0,
        bar_index=1,
        eligible_from_bar=2,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    assert ord1.status == OrderStatus.CANCELLED
    assert ord1.cancel_reason == "SUPERSEDED"
    assert ord2.status == OrderStatus.PENDING
    assert om.get_pending_entry_order("XAUUSD") is ord2


def test_expire_remaining_pending_orders_at_end_of_test():
    om = OrderManager()
    ts = pd.Timestamp("2026-01-01 00:00")
    ord_xau = om.create_order(
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1950.0,
        bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    om.expire_all_pending(timestamp=pd.Timestamp("2026-01-01 01:00"), bar_index=4, reason="END_OF_TEST")
    assert ord_xau.status == OrderStatus.EXPIRED
    assert ord_xau.cancel_reason == "END_OF_TEST"
    assert om.get_pending_entry_order("XAUUSD") is None

