from dataclasses import replace
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


def _make_quote(bid_close: float, ask_close: float, bar_index: int = 5) -> QuoteBar:
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


def test_reconcile_all_7_invariants_across_lifecycle_scenarios():
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0)

    # 1. Multiple closed trades -> flat at end
    acc1 = Account(initial_balance=10000.0)
    pf1 = Portfolio(account=acc1, instrument_config=inst)
    pf1.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, fee=2.0, bar_index=1))
    pf1.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.SELL, price=2015.0, fee=2.0, bar_index=2))
    pf1.on_fill(_make_fill("FILL-00003", "ORD-00003", OrderSide.SELL, price=2020.0, fee=2.0, bar_index=3))
    pf1.on_fill(_make_fill("FILL-00004", "ORD-00004", OrderSide.BUY, price=2005.0, fee=2.0, bar_index=4))
    rep1 = pf1.reconcile()
    assert rep1.is_valid is True
    assert rep1.errors == []

    # 2 & 3. Scale-in + Partial Close + Remaining Open Position
    acc2 = Account(initial_balance=10000.0)
    pf2 = Portfolio(account=acc2, instrument_config=inst, default_position_mode=PositionMode.NETTING_SCALE_IN)
    pf2.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.1, units=10.0, fee=1.5, bar_index=1))
    pf2.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.BUY, price=2010.0, quantity=0.2, units=20.0, fee=3.0, bar_index=2))
    pf2.on_fill(_make_fill("FILL-00003", "ORD-00003", OrderSide.SELL, price=2025.0, quantity=0.1, units=10.0, fee=1.5, bar_index=3))
    quote = _make_quote(bid_close=2030.0, ask_close=2031.0, bar_index=4)
    rep2 = pf2.reconcile({"XAUUSD": quote})
    assert rep2.is_valid is True
    assert rep2.errors == []

    # 4. Over-Close Reverse -> Remaining Short Open Position
    acc3 = Account(initial_balance=10000.0)
    pf3 = Portfolio(account=acc3, instrument_config=inst)
    pf3.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.1, units=10.0, fee=2.0, bar_index=1))
    pf3.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.SELL, price=2010.0, quantity=0.3, units=30.0, fee=6.0, bar_index=2))
    rep3 = pf3.reconcile({"XAUUSD": quote})
    assert rep3.is_valid is True
    assert rep3.errors == []


def test_reconcile_detects_corrupted_invariants_and_reports_detailed_errors():
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0)
    acc = Account(initial_balance=10000.0)
    pf = Portfolio(account=acc, instrument_config=inst, default_position_mode=PositionMode.NETTING_SCALE_IN)
    pf.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.2, units=20.0, fee=3.0, bar_index=1))
    pf.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.SELL, price=2010.0, quantity=0.1, units=10.0, fee=1.5, bar_index=2))

    # 5. Corrupt account.balance -> cash_invariant_ok == False and ledger_balance_ok == False
    orig_balance = acc.balance
    acc.balance += 50.0
    rep_cash = pf.reconcile({"XAUUSD": 2015.0})
    assert rep_cash.is_valid is False
    assert rep_cash.cash_invariant_ok is False
    assert any("INVARIANT_1_BALANCE" in err for err in rep_cash.errors)
    acc.balance = orig_balance

    # 6. Corrupt open_position_entry_fees -> fee_conservation_ok == False
    orig_open_fees = acc.open_position_entry_fees
    acc.open_position_entry_fees += 10.0
    rep_fee = pf.reconcile({"XAUUSD": 2015.0})
    assert rep_fee.is_valid is False
    assert rep_fee.fee_conservation_ok is False
    assert any("INVARIANT_3A_FEE_SPLIT" in err for err in rep_fee.errors)
    acc.open_position_entry_fees = orig_open_fees

    # 7 & 8. Corrupt TradeAllocation.allocated_units -> position_allocation_conservation_ok == False
    orig_alloc = pf.trade_allocations[0]
    pf.trade_allocations[0] = replace(orig_alloc, allocated_units=orig_alloc.allocated_units + 5.0)
    rep_alloc = pf.reconcile({"XAUUSD": 2015.0})
    assert rep_alloc.is_valid is False
    assert rep_alloc.position_allocation_conservation_ok is False
    assert any("INVARIANT_6_ENTRY_CONTRIBUTION" in err for err in rep_alloc.errors)
