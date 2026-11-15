"""
Unit tests for Task G2: Return Series, Log Returns, TWR, Wealth Index & Cumulative Return.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from quant_backtester.analytics.returns import (
    RETURN_SERIES_COLUMNS,
    compute_return_series,
    extract_active_period_returns,
)
from quant_backtester.portfolio.ledger import TransactionType


def _make_equity_df(equities: list[float]) -> pd.DataFrame:
    times = pd.date_range("2025-01-01 09:00:00", periods=len(equities), freq="1h")
    return pd.DataFrame({"time": times, "equity": equities})


def test_1_simple_return_computed_accurately():
    df = _make_equity_df([1000.0, 1100.0, 1045.0, 1149.5])
    res = compute_return_series(df)
    assert list(res.columns) == RETURN_SERIES_COLUMNS
    expected = [0.0, 0.10, -0.05, 0.10]
    for actual, exp in zip(res["simple_return"], expected):
        assert math.isclose(actual, exp, rel_tol=1e-9, abs_tol=1e-12)


def test_2_log_return_computed_accurately():
    df = _make_equity_df([1000.0, 1100.0, 1045.0])
    res = compute_return_series(df)
    assert math.isclose(res.loc[0, "log_return"], 0.0, abs_tol=1e-12)
    assert math.isclose(res.loc[1, "log_return"], math.log(1.10), rel_tol=1e-9)
    assert math.isclose(res.loc[2, "log_return"], math.log(0.95), rel_tol=1e-9)


def test_3_without_cash_flow_simple_return_equals_twr():
    df = _make_equity_df([1000.0, 1050.0, 1029.0, 1080.45])
    res = compute_return_series(df, initial_balance=1000.0)
    pd.testing.assert_series_equal(
        res["simple_return"], res["twr_return"], check_names=False
    )


def test_4_end_of_bar_deposit_does_not_distort_twr():
    # Bar 0: 1000 -> Bar 1: +10% trading gain (+100) + $500 deposit = 1600
    # Bar 2: +5% trading gain on 1600 (+80) = 1680
    eq_df = _make_equity_df([1000.0, 1600.0, 1680.0])
    ledger_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-01 10:00:00")],
            "transaction_type": [TransactionType.DEPOSIT.value],
            "amount": [500.0],
        }
    )
    res = compute_return_series(eq_df, ledger_df=ledger_df, initial_balance=1000.0)
    # Simple return at Bar 1 is +60%, but TWR return is (1600 - 500)/1000 - 1 = +10%!
    assert math.isclose(res.loc[1, "simple_return"], 0.60, rel_tol=1e-9)
    assert math.isclose(res.loc[1, "twr_return"], 0.10, rel_tol=1e-9)
    assert math.isclose(res.loc[2, "twr_return"], 0.05, rel_tol=1e-9)


def test_5_end_of_bar_withdrawal_does_not_distort_twr():
    # Bar 0: 1000 -> Bar 1: -5% trading loss (-50) - $200 withdrawal = 750
    # Bar 2: +10% trading gain on 750 (+75) = 825
    eq_df = _make_equity_df([1000.0, 750.0, 825.0])
    ledger_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-01 10:00:00")],
            "transaction_type": [TransactionType.WITHDRAWAL.value],
            "amount": [-200.0],
        }
    )
    res = compute_return_series(eq_df, ledger_df=ledger_df, initial_balance=1000.0)
    assert math.isclose(res.loc[1, "simple_return"], -0.25, rel_tol=1e-9)
    assert math.isclose(res.loc[1, "twr_return"], -0.05, rel_tol=1e-9)
    assert math.isclose(res.loc[2, "twr_return"], 0.10, rel_tol=1e-9)


def test_6_multiple_cash_flows_in_same_bar_twr():
    eq_df = _make_equity_df([1000.0, 1400.0])
    ledger_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2025-01-01 09:30:00"),
                pd.Timestamp("2025-01-01 10:00:00"),
            ],
            "transaction_type": [
                TransactionType.DEPOSIT.value,
                TransactionType.WITHDRAWAL.value,
            ],
            "amount": [500.0, -200.0],
        }
    )
    # Net CF in Bar 1 = +300 -> Trading equity = 1400 - 300 = 1100 -> TWR = +10%
    res = compute_return_series(eq_df, ledger_df=ledger_df, initial_balance=1000.0)
    assert math.isclose(res.loc[1, "external_cash_flow"], 300.0)
    assert math.isclose(res.loc[1, "twr_return"], 0.10, rel_tol=1e-9)


def test_7_wealth_index_progression():
    eq_df = _make_equity_df([1000.0, 1600.0, 1680.0])
    ledger_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-01 10:00:00")],
            "transaction_type": [TransactionType.DEPOSIT.value],
            "amount": [500.0],
        }
    )
    res = compute_return_series(eq_df, ledger_df=ledger_df, initial_balance=1000.0)
    assert math.isclose(res.loc[0, "wealth_index"], 1.0, rel_tol=1e-9)
    assert math.isclose(res.loc[1, "wealth_index"], 1.10, rel_tol=1e-9)
    assert math.isclose(res.loc[2, "wealth_index"], 1.155, rel_tol=1e-9)


def test_8_cumulative_return_matches_wealth_index_minus_one():
    df = _make_equity_df([1000.0, 1200.0, 900.0, 1080.0])
    res = compute_return_series(df, initial_balance=1000.0)
    for w, c in zip(res["wealth_index"], res["cumulative_return"]):
        assert math.isclose(c, w - 1.0, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(res.loc[3, "cumulative_return"], 0.08, rel_tol=1e-9)


def test_9_invalid_equity_or_unsorted_timestamps_handled_properly():
    # Zero / negative equity at bar 1 makes log_return NaN and subsequent bar return NaN
    df_zero = _make_equity_df([1000.0, 0.0, 0.0])
    res = compute_return_series(df_zero, initial_balance=1000.0)
    assert math.isclose(res.loc[1, "twr_return"], -1.0)
    assert np.isnan(res.loc[1, "log_return"])
    assert math.isclose(res.loc[1, "wealth_index"], 0.0)
    assert np.isnan(res.loc[2, "twr_return"])

    # Unsorted timestamps must raise ValueError
    bad_time_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-01")],
            "equity": [1000.0, 1010.0],
        }
    )
    with pytest.raises(ValueError, match="monotonic"):
        compute_return_series(bad_time_df)

    # Duplicate timestamps must raise ValueError
    dup_time_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "equity": [1000.0, 1010.0],
        }
    )
    with pytest.raises(ValueError, match="trùng lặp"):
        compute_return_series(dup_time_df)


def test_10_empty_and_single_observation_series_stable_schema():
    empty_res = compute_return_series(pd.DataFrame())
    assert empty_res.empty
    assert list(empty_res.columns) == RETURN_SERIES_COLUMNS
    assert extract_active_period_returns(empty_res).empty

    single_df = _make_equity_df([1000.0])
    single_res = compute_return_series(single_df, initial_balance=1000.0)
    assert len(single_res) == 1
    assert list(single_res.columns) == RETURN_SERIES_COLUMNS
    assert math.isclose(single_res.loc[0, "wealth_index"], 1.0)
    assert extract_active_period_returns(single_res, initial_balance=1000.0).empty
