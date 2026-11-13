import pandas as pd
import pytest

from quant_backtester.execution.slippage import FixedSlippageModel
from quant_backtester.execution.standard import StandardExecutionModel
from quant_backtester.models.order import ExecutionPhase, Order, OrderSide, OrderType
from quant_backtester.models.quote import QuoteBar


def _qb(bid_o=2000.0, bid_h=2010.0, bid_l=1990.0, bid_c=2005.0, spread=0.2) -> QuoteBar:
    return QuoteBar(
        symbol="XAUUSD",
        time=pd.Timestamp("2026-01-01 00:15"),
        bid_open=bid_o,
        bid_high=bid_h,
        bid_low=bid_l,
        bid_close=bid_c,
        ask_open=bid_o + spread,
        ask_high=bid_h + spread,
        ask_low=bid_l + spread,
        ask_close=bid_c + spread,
    )


def test_buy_and_sell_limit_in_range_and_no_adverse_slippage():
    # Even if slippage model has 0.50 slippage, LIMIT orders NEVER receive adverse slippage
    exec_model = StandardExecutionModel(slippage_model=FixedSlippageModel(0.50), limit_penetration=0.0)
    # Ask Open = 2000.2, Ask Low = 1990.2; Bid Open = 2000.0, Bid High = 2010.0
    qb = _qb(bid_o=2000.0, bid_h=2010.0, bid_l=1990.0, bid_c=2005.0, spread=0.2)

    # 1. BUY LIMIT at 1995.0: Ask Open (2000.2) > 1995.0, Ask Low (1990.2) <= 1995.0 -> fills in RANGE at 1995.0
    buy_lim = Order(
        order_id="ORD-L1",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1995.0,
        created_bar_index=0,
        eligible_from_bar=1,
    )
    assert exec_model.propose_entry_open(buy_lim, qb, bar_index=1) is None
    prop_b = exec_model.propose_entry_range(buy_lim, qb, bar_index=1)
    assert prop_b is not None
    assert prop_b.phase == ExecutionPhase.RANGE
    assert prop_b.fill_price == pytest.approx(1995.0)
    assert prop_b.slippage == 0.0

    # 2. SELL LIMIT at 2008.0: Bid Open (2000.0) < 2008.0, Bid High (2010.0) >= 2008.0 -> fills in RANGE at 2008.0
    sell_lim = Order(
        order_id="ORD-L2",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=2008.0,
        created_bar_index=0,
        eligible_from_bar=1,
    )
    assert exec_model.propose_entry_open(sell_lim, qb, bar_index=1) is None
    prop_s = exec_model.propose_entry_range(sell_lim, qb, bar_index=1)
    assert prop_s is not None
    assert prop_s.fill_price == pytest.approx(2008.0)
    assert prop_s.slippage == 0.0


def test_marketable_limit_at_open_and_limit_penetration():
    qb = _qb(bid_o=2000.0, bid_h=2010.0, bid_l=1990.0, bid_c=2005.0, spread=0.2)
    exec_model = StandardExecutionModel()

    # BUY LIMIT at 2005.0 when Ask Open is 2000.2 (favorable gap below limit) -> fills at Ask Open (2000.2)
    buy_gap = Order(
        order_id="ORD-LG1",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=2005.0,
        created_bar_index=0,
        eligible_from_bar=1,
    )
    prop_open = exec_model.propose_entry_open(buy_gap, qb, bar_index=1)
    assert prop_open is not None
    assert prop_open.phase == ExecutionPhase.OPEN
    assert prop_open.fill_price == pytest.approx(2000.2)

    # Penetration requirement: Ask Low = 1990.2. If BUY LIMIT = 1990.5 and limit_penetration = 0.5 -> requires Ask Low <= 1990.0 -> Not filled!
    exec_pen = StandardExecutionModel(limit_penetration=0.5)
    buy_pen = Order(
        order_id="ORD-LP1",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        units=1.0,
        limit_price=1990.5,
        created_bar_index=0,
        eligible_from_bar=1,
    )
    assert exec_pen.propose_entry_range(buy_pen, qb, bar_index=1) is None

