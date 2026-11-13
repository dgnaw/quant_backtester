import pandas as pd
import pytest

from quant_backtester.execution.standard import StandardExecutionModel
from quant_backtester.models.order import ExecutionPhase, Order, OrderOrigin, OrderSide, OrderType
from quant_backtester.models.quote import QuoteBar
from quant_backtester.portfolio.position import Position, PositionSide


def _qb(bar_time="2026-01-01 00:15", bid_o=2000.0, bid_h=2010.0, bid_l=1990.0, bid_c=2005.0, spread=0.2) -> QuoteBar:
    return QuoteBar(
        symbol="XAUUSD",
        time=pd.Timestamp(bar_time),
        bid_open=bid_o,
        bid_high=bid_h,
        bid_low=bid_l,
        bid_close=bid_c,
        ask_open=bid_o + spread,
        ask_high=bid_h + spread,
        ask_low=bid_l + spread,
        ask_close=bid_c + spread,
    )


def test_entry_order_ineligible_on_same_bar_as_signal():
    exec_model = StandardExecutionModel()
    qb = _qb()

    # Order created at bar 0 -> eligible_from_bar = 1
    order = Order(
        order_id="ORD-1",
        time=pd.Timestamp("2026-01-01 00:00"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        units=1.0,
        created_bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )

    # At bar_index = 0, order must NOT be proposed
    assert exec_model.propose_entry_open(order, qb, bar_index=0) is None
    assert exec_model.propose_entry_range(order, qb, bar_index=0) is None

    # At bar_index = 1, order IS eligible in Open phase
    prop = exec_model.propose_entry_open(order, qb, bar_index=1)
    assert prop is not None
    assert prop.phase == ExecutionPhase.OPEN


def test_position_opened_in_range_not_evaluated_for_sl_tp_in_same_range():
    exec_model = StandardExecutionModel()
    qb = _qb(bid_o=2000.0, bid_h=2020.0, bid_l=1970.0, bid_c=2005.0, spread=0.0)

    # Case 1: Position opened in OPEN phase of bar 1 -> eligible for SL/TP in RANGE phase of bar 1
    pos_open = Position(symbol="XAUUSD")
    pos_open.open(
        side=PositionSide.LONG,
        price=2000.0,
        units=1.0,
        timestamp=qb.time,
        stop_loss_price=1980.0,
        take_profit_price=2050.0,
        opened_bar_index=1,
        opened_phase=ExecutionPhase.OPEN,
    )
    prop_open = exec_model.propose_protective_range(pos_open, qb, bar_index=1)
    assert prop_open is not None
    assert prop_open.origin == OrderOrigin.PROTECTIVE_SL

    # Case 2: Position opened in RANGE phase of bar 1 (e.g. via Limit/Stop intrabar) -> NOT evaluated in bar 1 Range!
    pos_range = Position(symbol="XAUUSD")
    pos_range.open(
        side=PositionSide.LONG,
        price=1995.0,
        units=1.0,
        timestamp=qb.time,
        stop_loss_price=1980.0,
        take_profit_price=2010.0,
        opened_bar_index=1,
        opened_phase=ExecutionPhase.RANGE,
    )
    assert exec_model.propose_protective_range(pos_range, qb, bar_index=1) is None

    # On the next bar (bar_index = 2), pos_range IS evaluated for SL/TP
    prop_next_bar = exec_model.propose_protective_range(pos_range, qb, bar_index=2)
    assert prop_next_bar is not None

