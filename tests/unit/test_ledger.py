import pandas as pd
import pytest

from quant_backtester.config import InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.ledger import LEDGER_COLUMNS, Ledger, TransactionType
from quant_backtester.portfolio.portfolio import Portfolio


def _make_fill(
    fill_id: str,
    order_id: str,
    side: OrderSide,
    price: float,
    quantity: float = 0.1,
    units: float = 10.0,
    fee: float = 1.5,
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
        slippage=0.0,
        reason="TEST",
        metadata={"bar_index": bar_index},
    )


def test_ledger_deposit_withdrawal_and_open_fee_entry():
    acc = Account(initial_balance=10000.0)
    dep = acc.deposit(500.0, timestamp=pd.Timestamp("2026-01-01 00:00"), reason="INITIAL_TOPUP")
    assert dep.entry_id == "LED-00001"
    assert dep.transaction_type == TransactionType.DEPOSIT
    assert dep.amount == pytest.approx(500.0)
    assert dep.balance_after == pytest.approx(10500.0)

    wdr = acc.withdraw(500.0, timestamp=pd.Timestamp("2026-01-01 00:05"), reason="WITHDRAW")
    assert wdr.entry_id == "LED-00002"
    assert wdr.transaction_type == TransactionType.WITHDRAWAL
    assert wdr.amount == pytest.approx(-500.0)
    assert wdr.balance_after == pytest.approx(10000.0)

    pf = Portfolio(account=acc, instrument_config=InstrumentConfig(symbol="XAUUSD", contract_size=100.0))
    pf.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, fee=3.25, bar_index=1))

    assert len(acc.ledger.entries) == 3
    fee_entry = acc.ledger.entries[2]
    assert fee_entry.entry_id == "LED-00003"
    assert fee_entry.transaction_type == TransactionType.FEE
    assert fee_entry.amount == pytest.approx(-3.25)
    assert fee_entry.balance_after == pytest.approx(10000.0 - 3.25)
    assert fee_entry.order_id == "ORD-00001"
    assert fee_entry.fill_id == "FILL-00001"
    assert fee_entry.trade_id is None


def test_ledger_close_entries_no_double_counting_and_running_balance():
    acc = Account(initial_balance=10000.0)
    pf = Portfolio(account=acc, instrument_config=InstrumentConfig(symbol="XAUUSD", contract_size=100.0))

    pf.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.1, units=10.0, fee=2.0, bar_index=1))
    pf.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.SELL, price=2015.0, quantity=0.1, units=10.0, fee=2.5, bar_index=2))

    assert len(pf.trade_records) == 1
    tr = pf.trade_records[0]
    assert tr.gross_pnl == pytest.approx(150.0)
    assert tr.entry_fee == pytest.approx(2.0)
    assert tr.exit_fee == pytest.approx(2.5)
    assert tr.net_pnl == pytest.approx(145.5)

    # Ledger entries: 0=ENTRY_FEE(-2.0), 1=EXIT_FEE(-2.5), 2=TRADE_PNL(+150.0)
    entries = acc.ledger.entries
    assert len(entries) == 3
    assert entries[1].transaction_type == TransactionType.FEE
    assert entries[1].amount == pytest.approx(-2.5)
    assert entries[1].trade_id == tr.trade_id
    assert entries[2].transaction_type == TransactionType.TRADE_PNL
    assert entries[2].amount == pytest.approx(150.0)
    assert entries[2].trade_id == tr.trade_id

    # Running balance invariant
    assert entries[0].balance_after == pytest.approx(acc.initial_balance + entries[0].amount)
    for k in range(1, len(entries)):
        assert entries[k].balance_after == pytest.approx(entries[k - 1].balance_after + entries[k].amount)
    assert entries[-1].balance_after == pytest.approx(acc.balance)
    assert acc.initial_balance + sum(e.amount for e in entries) == pytest.approx(acc.balance)
    assert acc.balance == pytest.approx(10000.0 + 145.5)


def test_ledger_partial_close_and_scale_in_fee_reconciliation():
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0)
    acc = Account(initial_balance=10000.0)
    pf = Portfolio(account=acc, instrument_config=inst, default_position_mode=PositionMode.NETTING_SCALE_IN)

    pf.on_fill(_make_fill("FILL-00001", "ORD-00001", OrderSide.BUY, price=2000.0, quantity=0.1, units=10.0, fee=1.5, bar_index=1))
    pf.on_fill(_make_fill("FILL-00002", "ORD-00002", OrderSide.BUY, price=2010.0, quantity=0.2, units=20.0, fee=3.0, bar_index=2))

    # Partial close 0.1 lot (10 units out of 30 -> 1/3)
    pf.on_fill(_make_fill("FILL-00003", "ORD-00003", OrderSide.SELL, price=2020.0, quantity=0.1, units=10.0, fee=1.5, bar_index=3))
    # Remaining close 0.2 lot (20 units)
    pf.on_fill(_make_fill("FILL-00004", "ORD-00004", OrderSide.SELL, price=2025.0, quantity=0.2, units=20.0, fee=3.0, bar_index=4))

    assert len(pf.trade_records) == 2
    open_fee_entries = [e for e in acc.ledger.entries if e.transaction_type == TransactionType.FEE and e.trade_id is None]
    assert sum(-e.amount for e in open_fee_entries) == pytest.approx(sum(t.entry_fee for t in pf.trade_records))


def test_ledger_to_dataframe_schema_and_empty():
    empty_ledger = Ledger()
    df_empty = empty_ledger.to_dataframe()
    assert list(df_empty.columns) == LEDGER_COLUMNS
    assert df_empty.empty

    acc = Account(initial_balance=5000.0)
    acc.deposit(100.0, timestamp=pd.Timestamp("2026-01-01 00:00"), reason="TOPUP")
    df_pop = acc.ledger.to_dataframe()
    assert list(df_pop.columns) == LEDGER_COLUMNS
    assert len(df_pop) == 1
    assert df_pop.iloc[0]["entry_id"] == "LED-00001"
