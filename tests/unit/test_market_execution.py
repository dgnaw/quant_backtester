import pandas as pd
import pytest

from quant_backtester.execution.slippage import FixedSlippageModel
from quant_backtester.execution.standard import StandardExecutionModel
from quant_backtester.models.order import Order, OrderOrigin, OrderSide, OrderType
from quant_backtester.models.quote import QuoteBar


def test_market_buy_fills_at_ask_open_and_sell_fills_at_bid_open_with_slippage():
    # Bid Open = 1999.8, Ask Open = 2000.2 (Spread = 0.4)
    qb = QuoteBar(
        symbol="XAUUSD",
        time=pd.Timestamp("2026-01-01 00:15"),
        bid_open=1999.8,
        bid_high=2004.8,
        bid_low=1994.8,
        bid_close=2001.8,
        ask_open=2000.2,
        ask_high=2005.2,
        ask_low=1995.2,
        ask_close=2002.2,
    )
    exec_model = StandardExecutionModel(slippage_model=FixedSlippageModel(0.10))

    # BUY Market -> Base = Ask Open (2000.2) + Slippage (0.10) = 2000.3
    buy_ord = Order(
        order_id="ORD-M1",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        units=1.0,
        created_bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    prop_buy = exec_model.propose_entry_open(buy_ord, qb, bar_index=1)
    assert prop_buy is not None
    assert prop_buy.base_price == pytest.approx(2000.2)
    assert prop_buy.fill_price == pytest.approx(2000.3)
    assert prop_buy.slippage == pytest.approx(0.10)

    # SELL Market -> Base = Bid Open (1999.8) - Slippage (0.10) = 1999.7
    sell_ord = Order(
        order_id="ORD-M2",
        time=qb.time,
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.01,
        units=1.0,
        created_bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
    )
    prop_sell = exec_model.propose_entry_open(sell_ord, qb, bar_index=1)
    assert prop_sell is not None
    assert prop_sell.base_price == pytest.approx(1999.8)
    assert prop_sell.fill_price == pytest.approx(1999.7)
    assert prop_sell.slippage == pytest.approx(0.10)

