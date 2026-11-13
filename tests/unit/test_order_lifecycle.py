import pandas as pd
import pytest

from quant_backtester.execution.order_manager import OrderManager
from quant_backtester.models.order import OrderOrigin, OrderSide, OrderStatus, OrderType


def test_valid_order_lifecycle_transitions_and_event_history():
    om = OrderManager()
    ts0 = pd.Timestamp("2026-01-01 00:00")
    ts1 = pd.Timestamp("2026-01-01 00:15")

    # 1. CREATED -> PENDING -> FILLED
    order = om.create_order(
        time=ts0,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.05,
        units=5.0,
        bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    assert order.status == OrderStatus.PENDING
    # Event log recorded CREATED and PENDING
    events_for_ord = [e for e in om.events if e.order_id == order.order_id]
    assert [e.event_type for e in events_for_ord] == ["CREATED", "PENDING"]

    om.mark_filled(order, fill_price=2001.0, filled_quantity=0.05, timestamp=ts1, bar_index=1, reason="MARKET_FILL")
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == pytest.approx(0.05)
    assert order.remaining_quantity == pytest.approx(0.0)

    # Cannot transition a terminal FILLED order again
    with pytest.raises(ValueError, match="trạng thái kết thúc"):
        om.cancel_order(order, timestamp=ts1, bar_index=1, reason="USER_CANCELLED")
    with pytest.raises(ValueError, match="trạng thái kết thúc"):
        om.mark_filled(order, fill_price=2002.0, filled_quantity=0.05, timestamp=ts1, bar_index=1)


def test_created_to_rejected_and_pending_to_cancelled_or_expired():
    om = OrderManager()
    ts = pd.Timestamp("2026-01-01 00:00")

    # CREATED -> REJECTED
    rejected = om.create_rejected_order(
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1990.0,
        bar_index=0,
        reject_reason="RISK_REJECTED",
    )
    assert rejected.status == OrderStatus.REJECTED
    assert rejected.reject_reason == "RISK_REJECTED"
    with pytest.raises(ValueError, match="trạng thái kết thúc"):
        om.mark_filled(rejected, fill_price=1990.0, filled_quantity=0.01, timestamp=ts, bar_index=1)

    # PENDING -> CANCELLED
    to_cancel = om.create_order(
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1990.0,
        bar_index=0,
    )
    om.cancel_order(to_cancel, timestamp=ts, bar_index=1, reason="USER_CANCELLED")
    assert to_cancel.status == OrderStatus.CANCELLED
    assert to_cancel.cancel_reason == "USER_CANCELLED"

    # PENDING -> EXPIRED
    to_expire = om.create_order(
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.01,
        units=1.0,
        stop_price=1980.0,
        bar_index=0,
    )
    om.expire_order(to_expire, timestamp=ts, bar_index=2, reason="TIME_IN_FORCE_EXPIRED")
    assert to_expire.status == OrderStatus.EXPIRED
    assert len(om.events_to_dataframe()) >= 8

