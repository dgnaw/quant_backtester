import math
import pandas as pd
import pytest

from quant_backtester.config import InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.models.quote import QuoteBar
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.portfolio import Portfolio


def _make_fill(
    fill_id: str,
    order_id: str,
    side: OrderSide,
    price: float,
    quantity: float = 0.1,
    units: float = 10.0,
    fee: float = 1.5,
    slippage: float = 0.0,
    bar_index: int = 1,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        time=pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(minutes=15 * bar_index),
        symbol="XAUUSD",
        side=side,
        price=price,
        quantity=quantity,
        units=units,
        fee=fee,
        slippage=slippage,
        reason="TEST",
        metadata={"bar_index": bar_index, "slippage_cost": slippage * units},
    )


def _make_quote(bid_close: float, ask_close: float, bar_index: int = 2) -> QuoteBar:
    return QuoteBar(
        time=pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(minutes=15 * bar_index),
        symbol="XAUUSD",
        bid_open=bid_close,
        bid_high=bid_close + 1.0,
        bid_low=bid_close - 1.0,
        bid_close=bid_close,
        ask_open=ask_close,
        ask_high=ask_close + 1.0,
        ask_low=ask_close - 1.0,
        ask_close=ask_close,
    )


def test_account_initialization_valid_and_invalid():
    # 1. Valid initialization
    acc = Account(initial_balance=10000.0, account_currency="USD")
    assert acc.initial_balance == pytest.approx(10000.0)
    assert acc.balance == pytest.approx(10000.0)
    assert acc.equity == pytest.approx(10000.0)
    assert acc.unrealized_pnl == pytest.approx(0.0)
    assert acc.net_deposits == pytest.approx(0.0)
    assert acc.realized_gross_pnl == pytest.approx(0.0)
    assert acc.total_fees_paid == pytest.approx(0.0)
    assert acc.realized_fees == pytest.approx(0.0)
    assert acc.open_position_entry_fees == pytest.approx(0.0)
    assert acc.realized_net_pnl == pytest.approx(0.0)

    # 2. Invalid initial_balance <= 0 or NaN/inf raises ValueError
    for bad_val in (0.0, -500.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            Account(initial_balance=bad_val)


def test_entry_fee_deducted_immediately_and_tracked_in_open_position_entry_fees():
    # 3 & 4. Entry fee deducted immediately from balance and recorded in open_position_entry_fees
    acc = Account(initial_balance=10000.0)
    pf = Portfolio(account=acc, instrument_config=InstrumentConfig(symbol="XAUUSD", contract_size=100.0))

    f_open = _make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.1, units=10.0, fee=3.5)
    pf.on_fill(f_open, bar_index=1)

    assert acc.balance == pytest.approx(10000.0 - 3.5)
    assert acc.open_position_entry_fees == pytest.approx(3.5)
    assert acc.realized_fees == pytest.approx(0.0)
    assert acc.total_fees_paid == pytest.approx(3.5)
    assert acc.unrealized_pnl == pytest.approx(0.0)
    assert acc.equity == pytest.approx(acc.balance)


def test_long_and_short_mark_to_market_bid_ask_and_equity():
    # 5, 6, 7. LONG marked at bid_close, SHORT marked at ask_close, balance unchanged by mark price
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0)

    # LONG case
    acc_long = Account(initial_balance=10000.0)
    pf_long = Portfolio(account=acc_long, instrument_config=inst)
    pf_long.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.1, units=10.0, fee=2.0))

    quote = _make_quote(bid_close=2015.0, ask_close=2016.0)
    eq_long = pf_long.get_equity({"XAUUSD": quote})
    expected_unrl_long = (2015.0 - 2000.0) * 10.0  # 150.0
    assert acc_long.unrealized_pnl == pytest.approx(expected_unrl_long)
    assert acc_long.balance == pytest.approx(9998.0)
    assert eq_long == pytest.approx(9998.0 + 150.0)

    # SHORT case
    acc_short = Account(initial_balance=10000.0)
    pf_short = Portfolio(account=acc_short, instrument_config=inst)
    pf_short.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.SELL, price=2000.0, quantity=0.1, units=10.0, fee=2.0))

    eq_short = pf_short.get_equity({"XAUUSD": quote})
    expected_unrl_short = (2000.0 - 2016.0) * 10.0  # -160.0 (marked at ask_close=2016)
    assert acc_short.unrealized_pnl == pytest.approx(expected_unrl_short)
    assert acc_short.balance == pytest.approx(9998.0)
    assert eq_short == pytest.approx(9998.0 - 160.0)


def test_full_close_resets_unrealized_pnl_and_open_entry_fees_and_verifies_fee_identities():
    # 8, 9, 10, 11, 12. Full close profit & loss, fee identities
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0)
    acc = Account(initial_balance=10000.0)
    pf = Portfolio(account=acc, instrument_config=inst, default_position_mode=PositionMode.NETTING_SCALE_IN)

    # Open 0.1 lot @ 2000 (fee=2.0) + Scale-in 0.1 lot @ 2010 (fee=2.0) -> 20 units @ avg 2005, open_fee=4.0
    pf.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.1, units=10.0, fee=2.0, bar_index=1))
    pf.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.BUY, price=2010.0, quantity=0.1, units=10.0, fee=2.0, bar_index=2))
    assert acc.open_position_entry_fees == pytest.approx(4.0)
    assert acc.realized_fees == pytest.approx(0.0)
    assert acc.total_fees_paid == pytest.approx(4.0)

    # Partial close 50% (0.1 lot = 10 units) @ 2025 (exit_fee=2.5) -> gross = (2025 - 2005) * 10 = +200.0
    pf.on_fill(_make_fill("FILL-00003", "ORD-00003", OrderSide.SELL, price=2025.0, quantity=0.1, units=10.0, fee=2.5, bar_index=3))
    # Allocated entry fee = 2.0, exit fee = 2.5 -> realized_fees = 4.5, open_position_entry_fees = 2.0
    assert acc.open_position_entry_fees == pytest.approx(2.0)
    assert acc.realized_fees == pytest.approx(4.5)
    assert acc.total_fees_paid == pytest.approx(6.5)
    assert acc.total_fees_paid == pytest.approx(acc.realized_fees + acc.open_position_entry_fees)
    assert acc.realized_gross_pnl == pytest.approx(200.0)
    assert acc.realized_net_pnl == pytest.approx(200.0 - 4.5)

    # Close remaining 10 units at a loss @ 1995 (exit_fee=2.5) -> gross = (1995 - 2005) * 10 = -100.0
    pf.on_fill(_make_fill("FILL-00004", "ORD-00004", OrderSide.SELL, price=1995.0, quantity=0.1, units=10.0, fee=2.5, bar_index=4))
    assert acc.open_position_entry_fees == pytest.approx(0.0)
    assert acc.unrealized_pnl == pytest.approx(0.0)
    assert acc.equity == pytest.approx(acc.balance)
    assert acc.realized_gross_pnl == pytest.approx(100.0)
    assert acc.realized_fees == pytest.approx(9.0)
    assert acc.total_fees_paid == pytest.approx(9.0)
    assert acc.realized_net_pnl == pytest.approx(acc.realized_gross_pnl - acc.realized_fees)
    assert acc.balance == pytest.approx(10000.0 + 100.0 - 9.0)


def test_currency_compatibility_validation():
    acc = Account(initial_balance=10000.0, account_currency="USD")
    acc.validate_currency_compatibility("USD")
    with pytest.raises(ValueError, match="UNSUPPORTED_CURRENCY_CONVERSION"):
        acc.validate_currency_compatibility("JPY")
