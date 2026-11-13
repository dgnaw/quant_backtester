import pandas as pd
import pytest

from quant_backtester.data.trading_calendar import TradingCalendar
from quant_backtester.execution.order_manager import OrderManager
from quant_backtester.models.order import OrderOrigin, OrderSide, OrderStatus, OrderType, TimeInForce


def test_gtc_expire_after_bars_exact_ttl():
    """
    Ví dụ trong đặc tả D6:
    eligible_from_bar = 10, expire_after_bars = 3
    -> Order có hiệu lực trong bar 10, 11 và 12.
    -> Sau khi xử lý xong bar 12 mà chưa khớp thì chuyển sang EXPIRED.
    """
    om = OrderManager()
    ts = pd.Timestamp("2026-01-01 00:00")
    order = om.create_order(
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1950.0,
        bar_index=9,
        eligible_from_bar=10,
        time_in_force=TimeInForce.GTC,
        expire_after_bars=3,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )

    # Bar 10 and Bar 11 end: still PENDING
    om.check_expirations_after_bar(bar_index=10, timestamp=ts)
    assert order.status == OrderStatus.PENDING
    om.check_expirations_after_bar(bar_index=11, timestamp=ts)
    assert order.status == OrderStatus.PENDING

    # After Bar 12 finishes (3rd eligible bar: 10, 11, 12): must transition to EXPIRED
    om.check_expirations_after_bar(bar_index=12, timestamp=ts)
    assert order.status == OrderStatus.EXPIRED
    assert order.cancel_reason == "TIME_IN_FORCE_EXPIRED"


def test_ioc_order_cancelled_if_unfilled_in_eligible_bar():
    om = OrderManager()
    ts = pd.Timestamp("2026-01-01 00:00")
    order = om.create_order(
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1950.0,
        bar_index=0,
        eligible_from_bar=1,
        time_in_force=TimeInForce.IOC,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    om.check_expirations_after_bar(bar_index=1, timestamp=ts)
    assert order.status == OrderStatus.CANCELLED
    assert order.cancel_reason == "IOC_UNFILLED"


def test_day_order_expires_at_session_boundary():
    cal = TradingCalendar()
    om = OrderManager(calendar=cal)

    ts_day1 = pd.Timestamp("2026-01-01 23:45")
    ts_day2 = pd.Timestamp("2026-01-02 00:00")
    assert cal.session_id(ts_day1) != cal.session_id(ts_day2)

    order = om.create_order(
        time=ts_day1,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1950.0,
        bar_index=0,
        eligible_from_bar=1,
        time_in_force=TimeInForce.DAY,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )

    # Same day session -> stays PENDING
    om.check_expirations_after_bar(bar_index=1, timestamp=ts_day1, next_timestamp=ts_day1)
    assert order.status == OrderStatus.PENDING

    # Next bar crosses into new day session -> EXPIRED
    om.check_expirations_after_bar(bar_index=1, timestamp=ts_day1, next_timestamp=ts_day2)
    assert order.status == OrderStatus.EXPIRED
    assert order.cancel_reason == "DAY_SESSION_EXPIRED"

