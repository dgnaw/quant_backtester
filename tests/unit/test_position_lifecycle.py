import math
import pandas as pd
import pytest

from quant_backtester.config import InstrumentConfig
from quant_backtester.portfolio.position import Position, PositionSide, PositionView


def _inst(
    symbol: str = "XAUUSD",
    contract_size: float = 100.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    lot_step: float = 0.01,
) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol,
        contract_size=contract_size,
        min_lot=min_lot,
        max_lot=max_lot,
        lot_step=lot_step,
    )


def test_1_flat_to_long():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    assert pos.is_flat
    assert not pos.is_closed

    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.05,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=0.35,
        slippage_cost=0.50,
        order_id="ORD-00001",
        fill_id="FILL-00001",
        opened_bar_index=1,
    )
    assert pos.is_long
    assert not pos.is_flat
    assert pos.position_id.startswith("POS-")
    assert pos.quantity == pytest.approx(0.05)
    assert pos.units == pytest.approx(5.0)
    assert pos.average_entry_price == pytest.approx(2000.0)
    assert pos.entry_price == pytest.approx(2000.0)
    assert pos.entry_fee == pytest.approx(0.35)
    assert pos.entry_slippage_cost == pytest.approx(0.50)
    assert len(pos.contributions) == 1


def test_2_flat_to_short():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.SHORT,
        price=2010.0,
        quantity=0.02,
        timestamp=pd.Timestamp("2026-01-01 00:15"),
        fee=0.14,
    )
    assert pos.is_short
    assert pos.quantity == pytest.approx(0.02)
    assert pos.units == pytest.approx(2.0)
    assert pos.average_entry_price == pytest.approx(2010.0)
    assert pos.unrealized_pnl(1990.0) == pytest.approx(40.0)


def test_3_full_close_long():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.10,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=0.70,
        slippage_cost=1.00,
        order_id="ORD-00001",
        fill_id="FILL-00001",
        opened_bar_index=1,
    )
    trade = pos.close(
        exit_price=2020.0,
        timestamp=pd.Timestamp("2026-01-01 01:00"),
        fee=0.70,
        reason="TAKE_PROFIT",
        exit_order_id="ORD-00002",
        exit_fill_id="FILL-00002",
        exit_bar_index=5,
        exit_slippage_cost=0.0,
    )
    assert pos.is_flat
    assert pos.is_closed
    assert pos.quantity == 0.0
    assert pos.units == 0.0
    assert pos.entry_fee == 0.0
    assert pos.entry_slippage_cost == 0.0
    assert len(pos.contributions) == 0
    assert trade["gross_pnl"] == pytest.approx(200.0)
    assert trade["net_pnl"] == pytest.approx(200.0 - 1.40)


def test_4_full_close_short():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.SHORT,
        price=2000.0,
        quantity=0.10,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=0.50,
    )
    trade = pos.close(
        exit_price=1980.0,
        timestamp=pd.Timestamp("2026-01-01 01:00"),
        fee=0.50,
        reason="TAKE_PROFIT",
    )
    assert pos.is_flat
    assert pos.is_closed
    assert trade["gross_pnl"] == pytest.approx(200.0)
    assert trade["net_pnl"] == pytest.approx(199.0)


def test_5_weighted_average_price_on_scale_in():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.02,  # 2 units
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=0.20,
    )
    pos.scale_in(
        side=PositionSide.LONG,
        price=2050.0,
        quantity=0.03,  # 3 units
        timestamp=pd.Timestamp("2026-01-01 00:15"),
        fee=0.30,
    )
    # (2000 * 2 + 2050 * 3) / 5 = 10150 / 5 = 2030.0
    assert pos.average_entry_price == pytest.approx(2030.0)
    assert pos.entry_price == pytest.approx(2030.0)


def test_6_scale_in_adds_quantity_and_units():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(side=PositionSide.LONG, price=2000.0, quantity=0.02, timestamp=pd.Timestamp("2026-01-01 00:00"))
    orig_id = pos.position_id
    pos.scale_in(side=PositionSide.LONG, price=2010.0, quantity=0.04, timestamp=pd.Timestamp("2026-01-01 00:15"))
    assert pos.position_id == orig_id
    assert pos.quantity == pytest.approx(0.06)
    assert pos.units == pytest.approx(6.0)
    assert len(pos.contributions) == 2


def test_7_scale_in_adds_fee_and_slippage_cost():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.02,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=0.14,
        slippage_cost=0.20,
    )
    pos.scale_in(
        side=PositionSide.LONG,
        price=2010.0,
        quantity=0.03,
        timestamp=pd.Timestamp("2026-01-01 00:15"),
        fee=0.21,
        slippage_cost=0.30,
    )
    assert pos.entry_fee == pytest.approx(0.35)
    assert pos.entry_slippage_cost == pytest.approx(0.50)


def test_8_partial_close_allocates_entry_fee():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.10,  # 10 units
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=1.00,
    )
    # Close 40% (4.0 units = 0.04 lot)
    trade = pos.close(
        exit_price=2010.0,
        timestamp=pd.Timestamp("2026-01-01 00:30"),
        fee=0.40,
        close_units=4.0,
    )
    assert not pos.is_flat
    assert not pos.is_closed
    assert trade["entry_fee"] == pytest.approx(0.40)
    assert pos.entry_fee == pytest.approx(0.60)
    assert pos.quantity == pytest.approx(0.06)
    assert pos.units == pytest.approx(6.0)
    assert pos.average_entry_price == pytest.approx(2000.0)


def test_9_partial_close_allocates_entry_slippage():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.10,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=1.00,
        slippage_cost=2.50,
    )
    trade = pos.close(
        exit_price=2010.0,
        timestamp=pd.Timestamp("2026-01-01 00:30"),
        fee=0.40,
        close_units=4.0,
    )
    assert trade["entry_slippage_cost"] == pytest.approx(1.00)
    assert pos.entry_slippage_cost == pytest.approx(1.50)


def test_10_multiple_sequential_partial_closes():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.10,  # 10 units
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=1.00,
        slippage_cost=2.00,
    )
    t1 = pos.close(exit_price=2010.0, timestamp=pd.Timestamp("2026-01-01 00:15"), fee=0.30, close_units=3.0)
    t2 = pos.close(exit_price=2020.0, timestamp=pd.Timestamp("2026-01-01 00:30"), fee=0.30, close_units=3.0)
    t3 = pos.close(exit_price=2030.0, timestamp=pd.Timestamp("2026-01-01 00:45"), fee=0.40, close_units=4.0)

    assert pos.is_flat
    assert pos.is_closed
    assert t1["entry_fee"] + t2["entry_fee"] + t3["entry_fee"] == pytest.approx(1.00)
    assert t1["entry_slippage_cost"] + t2["entry_slippage_cost"] + t3["entry_slippage_cost"] == pytest.approx(2.00)


def test_11_full_close_leaves_zero_residual_allocation():
    inst = _inst()
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    pos.open(
        side=PositionSide.LONG,
        price=2000.0,
        quantity=0.03,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        fee=0.333333333,
        slippage_cost=0.777777777,
    )
    t1 = pos.close(exit_price=2005.0, timestamp=pd.Timestamp("2026-01-01 00:15"), fee=0.10, close_units=1.0)
    t2 = pos.close(exit_price=2010.0, timestamp=pd.Timestamp("2026-01-01 00:30"), fee=0.20, close_units=2.0)

    assert pos.is_closed
    assert pos.entry_fee == 0.0
    assert pos.entry_slippage_cost == 0.0
    assert t1["entry_fee"] + t2["entry_fee"] == pytest.approx(0.333333333, rel=1e-12)
    assert t1["entry_slippage_cost"] + t2["entry_slippage_cost"] == pytest.approx(0.777777777, rel=1e-12)


def test_12_cannot_close_already_closed_or_flat_position():
    pos = Position(symbol="XAUUSD", instrument_config=_inst())
    with pytest.raises(ValueError):
        pos.close(exit_price=2000.0, timestamp=pd.Timestamp("2026-01-01 00:00"))

    pos.open(side=PositionSide.LONG, price=2000.0, quantity=0.01, timestamp=pd.Timestamp("2026-01-01 00:00"))
    pos.close(exit_price=2005.0, timestamp=pd.Timestamp("2026-01-01 00:15"))
    with pytest.raises(ValueError):
        pos.close(exit_price=2005.0, timestamp=pd.Timestamp("2026-01-01 00:30"))


def test_13_cannot_open_or_scale_in_with_invalid_values():
    pos = Position(symbol="XAUUSD", instrument_config=_inst())
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.FLAT, price=2000.0, quantity=0.01, timestamp=pd.Timestamp("2026-01-01"))
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.LONG, price=0.0, quantity=0.01, timestamp=pd.Timestamp("2026-01-01"))
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.LONG, price=2000.0, quantity=-0.01, timestamp=pd.Timestamp("2026-01-01"))
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.LONG, price=2000.0, quantity=0.01, timestamp=pd.Timestamp("2026-01-01"), fee=-0.1)


def test_14_rejects_nan_inf_and_invalid_lot_step():
    inst = _inst(min_lot=0.01, max_lot=5.0, lot_step=0.01)
    pos = Position(symbol="XAUUSD", instrument_config=inst)
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.LONG, price=math.nan, quantity=0.01, timestamp=pd.Timestamp("2026-01-01"))
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.LONG, price=math.inf, quantity=0.01, timestamp=pd.Timestamp("2026-01-01"))
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.LONG, price=2000.0, quantity=0.015, timestamp=pd.Timestamp("2026-01-01"))
    with pytest.raises(ValueError):
        pos.open(side=PositionSide.LONG, price=2000.0, quantity=10.0, timestamp=pd.Timestamp("2026-01-01"))


def test_15_position_view_is_immutable():
    pos = Position(symbol="XAUUSD", instrument_config=_inst())
    pos.open(side=PositionSide.LONG, price=2000.0, quantity=0.02, timestamp=pd.Timestamp("2026-01-01 00:00"))
    view: PositionView = pos.to_view()
    assert view.quantity == pytest.approx(0.02)
    assert view.units == pytest.approx(2.0)
    with pytest.raises((AttributeError, TypeError)):
        view.quantity = 99.0  # type: ignore[misc]


def test_16_new_position_lifecycle_gets_new_position_id():
    inst = _inst()
    pos1 = Position(symbol="XAUUSD", instrument_config=inst)
    pos1.open(side=PositionSide.LONG, price=2000.0, quantity=0.01, timestamp=pd.Timestamp("2026-01-01 00:00"))
    id1 = pos1.position_id
    pos1.close(exit_price=2005.0, timestamp=pd.Timestamp("2026-01-01 00:15"))

    # A closed Position cannot be reopened; a new Position lifecycle gets a distinct position_id
    with pytest.raises(ValueError):
        pos1.open(side=PositionSide.LONG, price=2010.0, quantity=0.01, timestamp=pd.Timestamp("2026-01-01 00:30"))

    pos2 = Position(symbol="XAUUSD", instrument_config=inst)
    pos2.open(side=PositionSide.LONG, price=2010.0, quantity=0.01, timestamp=pd.Timestamp("2026-01-01 00:30"))
    assert pos2.position_id != id1
