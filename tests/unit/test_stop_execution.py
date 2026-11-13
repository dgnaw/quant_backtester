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


def test_buy_and_sell_stop_market_in_range_and_gap_at_open():
    exec_model = StandardExecutionModel(slippage_model=FixedSlippageModel(0.10))
    # Ask Open = 2000.2, Ask High = 2010.2; Bid Open = 2000.0, Bid Low = 1990.0
    qb = _qb()

    # 1. BUY STOP_MARKET at 2005.0 (in Range): Ask High (2010.2) >= 2005.0 -> Base = 2005.0 + 0.10 = 2005.10
    buy_stop = Order(
        order_id="ORD-S1",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_MARKET,
        quantity=0.01,
        units=1.0,
        stop_price=2005.0,
        created_bar_index=0,
        eligible_from_bar=1,
    )
    assert exec_model.propose_entry_open(buy_stop, qb, bar_index=1) is None
    prop_bs = exec_model.propose_entry_range(buy_stop, qb, bar_index=1)
    assert prop_bs is not None
    assert prop_bs.phase == ExecutionPhase.RANGE
    assert prop_bs.base_price == pytest.approx(2005.0)
    assert prop_bs.fill_price == pytest.approx(2005.10)

    # 2. SELL STOP_MARKET at 1995.0 (in Range): Bid Low (1990.0) <= 1995.0 -> Base = 1995.0 - 0.10 = 1994.90
    sell_stop = Order(
        order_id="ORD-S2",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.01,
        units=1.0,
        stop_price=1995.0,
        created_bar_index=0,
        eligible_from_bar=1,
    )
    assert exec_model.propose_entry_open(sell_stop, qb, bar_index=1) is None
    prop_ss = exec_model.propose_entry_range(sell_stop, qb, bar_index=1)
    assert prop_ss is not None
    assert prop_ss.base_price == pytest.approx(1995.0)
    assert prop_ss.fill_price == pytest.approx(1994.90)

    # 3. BUY STOP_MARKET Gap at Open: Stop = 1998.0, Ask Open = 2000.2 >= 1998.0 -> Base = Ask Open (2000.2) + 0.10 = 2000.30
    buy_stop_gap = Order(
        order_id="ORD-SG1",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_MARKET,
        quantity=0.01,
        units=1.0,
        stop_price=1998.0,
        created_bar_index=0,
        eligible_from_bar=1,
    )
    prop_gap = exec_model.propose_entry_open(buy_stop_gap, qb, bar_index=1)
    assert prop_gap is not None
    assert prop_gap.phase == ExecutionPhase.OPEN
    assert prop_gap.base_price == pytest.approx(2000.2)
    assert prop_gap.fill_price == pytest.approx(2000.30)

