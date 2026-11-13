import pandas as pd
import pytest

from quant_backtester.models.order import (
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    validate_and_normalize_lot,
)


def test_order_types_and_stop_market_alias():
    assert OrderType.STOP == OrderType.STOP_MARKET

    # 1. MARKET order defaults to IOC and requires no limit/stop price
    ts = pd.Timestamp("2026-01-01 00:00")
    m_order = Order(
        order_id="ORD-001",
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.05,
        units=5.0,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    assert m_order.time_in_force == TimeInForce.IOC
    assert m_order.filled_quantity == 0.0
    assert m_order.remaining_quantity == pytest.approx(0.05)

    # 2. LIMIT order defaults to GTC and requires limit_price
    l_order = Order(
        order_id="ORD-002",
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.10,
        units=10.0,
        limit_price=1990.0,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    assert l_order.time_in_force == TimeInForce.GTC
    assert l_order.limit_price == pytest.approx(1990.0)

    # 3. STOP_MARKET order defaults to GTC and requires stop_price
    s_order = Order(
        order_id="ORD-003",
        time=ts,
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.10,
        units=10.0,
        stop_price=1980.0,
        origin=OrderOrigin.PROTECTIVE_SL,
    )
    assert s_order.time_in_force == TimeInForce.GTC
    assert s_order.stop_price == pytest.approx(1980.0)


def test_order_validation_missing_price_or_invalid_quantity():
    ts = pd.Timestamp("2026-01-01 00:00")

    # LIMIT without limit_price
    with pytest.raises(ValueError, match="limit_price"):
        Order(
            order_id="ORD-BAD1",
            time=ts,
            symbol="XAUUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            units=1.0,
        )

    # STOP_MARKET without stop_price
    with pytest.raises(ValueError, match="stop_price"):
        Order(
            order_id="ORD-BAD2",
            time=ts,
            symbol="XAUUSD",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_MARKET,
            quantity=0.01,
            units=1.0,
        )

    # Non-positive quantity
    with pytest.raises(ValueError, match="quantity"):
        Order(
            order_id="ORD-BAD3",
            time=ts,
            symbol="XAUUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.0,
            units=0.0,
        )


def test_validate_and_normalize_lot_with_decimal_precision():
    # Valid lot sizes including floating point representation like 0.07
    assert validate_and_normalize_lot(0.01, min_lot=0.01, max_lot=100.0, lot_step=0.01) == pytest.approx(0.01)
    assert validate_and_normalize_lot(0.07, min_lot=0.01, max_lot=100.0, lot_step=0.01) == pytest.approx(0.07)
    assert validate_and_normalize_lot(0.01 + 0.02, min_lot=0.01, max_lot=100.0, lot_step=0.01) == pytest.approx(0.03)

    # Invalid step
    with pytest.raises(ValueError, match="lot_step"):
        validate_and_normalize_lot(0.015, min_lot=0.01, max_lot=100.0, lot_step=0.01)

    # Below min_lot / above max_lot
    with pytest.raises(ValueError, match="min_lot"):
        validate_and_normalize_lot(0.005, min_lot=0.01, max_lot=100.0, lot_step=0.01)
    with pytest.raises(ValueError, match="max_lot"):
        validate_and_normalize_lot(150.0, min_lot=0.01, max_lot=100.0, lot_step=0.01)

