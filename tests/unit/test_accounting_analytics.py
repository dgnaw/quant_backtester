"""
Unit tests for Task G1: Ledger Cash-Flow Accounting & Gate 03 Report Reconciliation.
"""

from __future__ import annotations

import math
import pandas as pd
import pytest

from quant_backtester.analytics.accounting import (
    map_cash_flows_to_bars,
    reconcile_report_accounting,
    summarize_ledger_cash_flows,
)
from quant_backtester.config import InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.models.quote import QuoteBar
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.ledger import TransactionType
from quant_backtester.portfolio.portfolio import Portfolio


def test_1_initial_balance_not_counted_as_external_cash_flow():
    acc = Account(initial_balance=5000.0)
    # Even if an explicit INITIAL_BALANCE ledger entry is recorded, it must not be treated as an external deposit
    acc.ledger.record(
        pd.Timestamp("2025-01-01 00:00:00"),
        TransactionType.DEPOSIT,
        5000.0,
        5000.0,
        reason="INITIAL_BALANCE",
    )
    summary = summarize_ledger_cash_flows(acc.ledger.to_dataframe(), initial_balance=5000.0)
    assert math.isclose(summary.initial_balance, 5000.0)
    assert math.isclose(summary.total_deposits, 0.0)
    assert math.isclose(summary.total_withdrawals, 0.0)
    assert math.isclose(summary.net_external_cash_flow, 0.0)
    assert summary.cash_flows_by_time.empty


def test_2_mid_period_deposit_recorded_in_cash_flow_summary():
    acc = Account(initial_balance=1000.0)
    acc.deposit(500.0, pd.Timestamp("2025-01-02 10:00:00"), reason="TOP_UP")
    summary = summarize_ledger_cash_flows(acc.ledger.to_dataframe(), initial_balance=1000.0)
    assert math.isclose(summary.total_deposits, 500.0)
    assert math.isclose(summary.total_withdrawals, 0.0)
    assert math.isclose(summary.net_external_cash_flow, 500.0)
    assert math.isclose(summary.final_ledger_balance, 1500.0)


def test_3_mid_period_withdrawal_recorded_in_cash_flow_summary():
    acc = Account(initial_balance=2000.0)
    acc.withdraw(400.0, pd.Timestamp("2025-01-02 10:00:00"), reason="PAYOUT")
    summary = summarize_ledger_cash_flows(acc.ledger.to_dataframe(), initial_balance=2000.0)
    assert math.isclose(summary.total_deposits, 0.0)
    assert math.isclose(summary.total_withdrawals, 400.0)
    assert math.isclose(summary.net_external_cash_flow, -400.0)
    assert math.isclose(summary.final_ledger_balance, 1600.0)


def test_4_multiple_cash_flows_in_same_bar_aggregated_correctly():
    acc = Account(initial_balance=1000.0)
    acc.deposit(600.0, pd.Timestamp("2025-01-02 10:00:00"))
    acc.withdraw(150.0, pd.Timestamp("2025-01-02 10:00:00"))
    acc.deposit(50.0, pd.Timestamp("2025-01-02 10:05:00"))
    summary = summarize_ledger_cash_flows(acc.ledger.to_dataframe(), initial_balance=1000.0)

    bar_times = pd.Series(
        [
            pd.Timestamp("2025-01-02 09:45:00"),
            pd.Timestamp("2025-01-02 10:00:00"),
            pd.Timestamp("2025-01-02 10:15:00"),
        ]
    )
    cf_bars = map_cash_flows_to_bars(bar_times, summary.cash_flows_by_time)
    assert list(cf_bars) == [0.0, 450.0, 50.0]


def test_5_gate03_reconciliation_with_closed_trades():
    acc = Account(initial_balance=1000.0)
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10)
    pf = Portfolio(account=acc, instrument_config=inst)

    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-02 09:00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.10, 10.0, fee=2.0)
    )
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2025-01-02 10:00:00"), "XAUUSD", OrderSide.SELL, 2015.0, 0.10, 10.0, fee=2.0)
    )

    eq_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:00:00")],
            "balance": [998.0, 1146.0],
            "equity": [998.0, 1146.0],
        }
    )
    rec = reconcile_report_accounting(
        equity_df=eq_df,
        trades_df=pf.trades_to_dataframe(),
        ledger_df=acc.ledger.to_dataframe(),
        open_positions_df=pf.open_positions_to_dataframe(),
        initial_balance=1000.0,
    )
    assert rec.is_valid
    assert math.isclose(rec.actual_value, 146.0)
    assert math.isclose(rec.expected_value, 146.0)
    assert math.isclose(rec.difference, 0.0, abs_tol=1e-12)


def test_6_gate03_reconciliation_with_open_positions_at_end_of_test():
    acc = Account(initial_balance=1000.0)
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10)
    pf = Portfolio(account=acc, instrument_config=inst)

    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-02 09:00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.10, 10.0, fee=3.0)
    )
    qb = QuoteBar(
        time=pd.Timestamp("2025-01-02 10:00:00"),
        symbol="XAUUSD",
        bid_open=2020.0,
        bid_high=2025.0,
        bid_low=2018.0,
        bid_close=2022.0,
        ask_open=2021.0,
        ask_high=2026.0,
        ask_low=2019.0,
        ask_close=2023.0,
    )
    eq_val = pf.get_equity({"XAUUSD": qb})
    open_df = pf.open_positions_to_dataframe({"XAUUSD": qb})
    eq_df = pd.DataFrame(
        {"time": [qb.time], "balance": [acc.balance], "equity": [eq_val]}
    )

    rec = reconcile_report_accounting(
        equity_df=eq_df,
        trades_df=pf.trades_to_dataframe(),
        ledger_df=acc.ledger.to_dataframe(),
        open_positions_df=open_df,
        initial_balance=1000.0,
    )
    assert rec.is_valid
    # Unrealized PnL = (2022 - 2000)*10 = +220.0, remaining entry fee = 3.0 -> Net Profit = +217.0
    assert math.isclose(rec.actual_value, 217.0)
    assert math.isclose(rec.expected_value, 217.0)


def test_7_gate03_reconciliation_with_partial_close_and_remaining_entry_fee():
    acc = Account(initial_balance=2000.0)
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10)
    pf = Portfolio(account=acc, instrument_config=inst, default_position_mode=PositionMode.NETTING_SCALE_IN)

    # Open 0.20 lots (20 units) @ 2000, fee = 6.0
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-02 09:00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.20, 20.0, fee=6.0)
    )
    # Deposit 500 mid-run
    acc.deposit(500.0, pd.Timestamp("2025-01-02 09:30:00"))
    # Partial close 0.10 lots (10 units) @ 2010, exit fee = 3.0 (allocates 3.0 entry fee; remaining entry fee = 3.0)
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2025-01-02 10:00:00"), "XAUUSD", OrderSide.SELL, 2010.0, 0.10, 10.0, fee=3.0)
    )
    qb = QuoteBar(
        time=pd.Timestamp("2025-01-02 11:00:00"),
        symbol="XAUUSD",
        bid_open=2015.0,
        bid_high=2020.0,
        bid_low=2012.0,
        bid_close=2018.0,
        ask_open=2016.0,
        ask_high=2021.0,
        ask_low=2013.0,
        ask_close=2019.0,
    )
    eq_val = pf.get_equity({"XAUUSD": qb})
    open_df = pf.open_positions_to_dataframe({"XAUUSD": qb})
    eq_df = pd.DataFrame({"time": [qb.time], "balance": [acc.balance], "equity": [eq_val]})

    rec = reconcile_report_accounting(
        equity_df=eq_df,
        trades_df=pf.trades_to_dataframe(),
        ledger_df=acc.ledger.to_dataframe(),
        open_positions_df=open_df,
        initial_balance=2000.0,
    )
    assert rec.is_valid
    # Closed trade net_pnl = (2010-2000)*10 - 3 - 3 = +94.0
    # Open 10 units unrealized_pnl = (2018-2000)*10 = +180.0, remaining_entry_fee = 3.0
    # Total net trading profit = 94.0 + 180.0 - 3.0 = 271.0
    assert math.isclose(rec.actual_value, 271.0)
    assert math.isclose(rec.expected_value, 271.0)


def test_8_discrepancy_between_equity_and_trades_is_detected():
    eq_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:00:00")],
            "balance": [1000.0, 1200.0],
            "equity": [1000.0, 1200.0],
        }
    )
    trades_df = pd.DataFrame({"net_pnl": [150.0], "gross_pnl": [155.0], "total_fee": [5.0]})
    rec = reconcile_report_accounting(
        equity_df=eq_df,
        trades_df=trades_df,
        initial_balance=1000.0,
    )
    assert not rec.is_valid
    assert "REPORT_NET_PROFIT_CONSERVATION" in rec.checks_failed
    assert math.isclose(rec.difference, 50.0)


def test_9_reconciliation_exact_when_no_external_cash_flows():
    eq_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:00:00")],
            "balance": [1000.0, 1085.0],
            "equity": [1000.0, 1085.0],
        }
    )
    trades_df = pd.DataFrame({"net_pnl": [85.0], "gross_pnl": [90.0], "total_fee": [5.0]})
    rec = reconcile_report_accounting(
        {"equity": eq_df, "trades": trades_df, "initial_balance": 1000.0}
    )
    assert rec.is_valid
    assert math.isclose(rec.actual_value, 85.0)
    assert math.isclose(rec.expected_value, 85.0)


def test_10_empty_schemas_handled_gracefully():
    summary = summarize_ledger_cash_flows(None, initial_balance=1000.0)
    assert summary.net_external_cash_flow == 0.0
    rec = reconcile_report_accounting(
        equity_df=pd.DataFrame(),
        trades_df=pd.DataFrame(),
        ledger_df=pd.DataFrame(),
        open_positions_df=pd.DataFrame(),
        initial_balance=1000.0,
    )
    assert rec.is_valid
    assert rec.actual_value == 0.0
    assert rec.expected_value == 0.0
    with pytest.raises(ValueError):
        summarize_ledger_cash_flows(None, initial_balance=-100.0)
