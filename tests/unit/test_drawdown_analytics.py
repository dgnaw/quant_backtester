"""
Unit tests for Task G3: Drawdown Curve, Underwater Streaks, Drawdown Episodes & Summary Metrics.
"""

from __future__ import annotations

import math
import pandas as pd

from quant_backtester.analytics.drawdown import (
    DRAWDOWN_EPISODE_COLUMNS,
    DRAWDOWN_SERIES_COLUMNS,
    compute_drawdown_series,
    extract_drawdown_episodes,
    summarize_drawdown,
)
from quant_backtester.portfolio.ledger import TransactionType


def _make_eq(equities: list[float]) -> pd.DataFrame:
    times = pd.date_range("2025-01-01 00:00:00", periods=len(equities), freq="1h")
    return pd.DataFrame({"time": times, "equity": equities})


def test_1_hand_calculated_drawdown_series():
    df = _make_eq([1000.0, 1100.0, 990.0, 1045.0, 1100.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    assert list(dd.columns) == DRAWDOWN_SERIES_COLUMNS
    # Bar 2: 990 / 1100 - 1 = -10% (-110 USD)
    assert math.isclose(dd.loc[2, "drawdown_pct"], -10.0, rel_tol=1e-9)
    assert math.isclose(dd.loc[2, "drawdown_depth_pct"], 10.0, rel_tol=1e-9)
    assert math.isclose(dd.loc[2, "account_drawdown_amount"], -110.0, rel_tol=1e-9)
    assert math.isclose(dd.loc[2, "account_drawdown_depth"], 110.0, rel_tol=1e-9)
    # Bar 3: 1045 / 1100 - 1 = -5% (-55 USD)
    assert math.isclose(dd.loc[3, "drawdown_pct"], -5.0, rel_tol=1e-9)


def test_2_peak_and_valley_tie_breaking_rules():
    # Two equal peaks at index 1 and 2 (1100.0) before drop; two equal valleys at index 3 and 4 (990.0)
    df = _make_eq([1000.0, 1100.0, 1100.0, 990.0, 990.0, 1100.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    eps = extract_drawdown_episodes(dd)
    assert len(eps) == 1
    ep = eps.iloc[0]
    # Latest peak bar before decline is index 2; first bar at deepest valley is index 3
    assert int(ep["peak_bar_index"]) == 2
    assert int(ep["valley_bar_index"]) == 3
    assert int(ep["recovery_bar_index"]) == 5


def test_3_recovered_episode_attributes():
    df = _make_eq([1000.0, 1200.0, 1080.0, 960.0, 1080.0, 1250.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    eps = extract_drawdown_episodes(dd)
    assert len(eps) == 1
    ep = eps.iloc[0]
    assert bool(ep["is_recovered"]) is True
    assert int(ep["peak_bar_index"]) == 1
    assert int(ep["valley_bar_index"]) == 3
    assert int(ep["recovery_bar_index"]) == 5
    assert int(ep["decline_bars"]) == 2
    assert int(ep["recovery_bars"]) == 2
    assert int(ep["duration_bars"]) == 3  # Bars 2, 3, 4 are strictly underwater
    assert math.isclose(float(ep["max_drawdown_pct"]), -20.0, rel_tol=1e-9)
    assert math.isclose(float(ep["max_drawdown_amount"]), 240.0, rel_tol=1e-9)


def test_4_unrecovered_episode_at_end_of_test():
    df = _make_eq([1000.0, 1200.0, 1080.0, 900.0, 950.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    eps = extract_drawdown_episodes(dd)
    assert len(eps) == 1
    ep = eps.iloc[0]
    assert bool(ep["is_recovered"]) is False
    assert ep["recovery_time"] is None
    assert ep["recovery_bar_index"] is None
    assert ep["recovery_bars"] is None
    assert int(ep["peak_bar_index"]) == 1
    assert int(ep["valley_bar_index"]) == 3
    assert int(ep["duration_bars"]) == 3  # Bars 2, 3, 4


def test_5_multiple_non_overlapping_episodes():
    df = _make_eq([1000.0, 900.0, 1000.0, 1100.0, 880.0, 990.0, 1150.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    eps = extract_drawdown_episodes(dd)
    assert len(eps) == 2
    assert int(eps.iloc[0]["episode_id"]) == 1
    assert int(eps.iloc[0]["peak_bar_index"]) == 0
    assert int(eps.iloc[0]["recovery_bar_index"]) == 2
    assert math.isclose(float(eps.iloc[0]["max_drawdown_pct"]), -10.0)

    assert int(eps.iloc[1]["episode_id"]) == 2
    assert int(eps.iloc[1]["peak_bar_index"]) == 3
    assert int(eps.iloc[1]["recovery_bar_index"]) == 6
    assert math.isclose(float(eps.iloc[1]["max_drawdown_pct"]), -20.0)


def test_6_underwater_bars_counter():
    df = _make_eq([1000.0, 950.0, 900.0, 980.0, 1000.0, 1050.0, 1020.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    assert list(dd["underwater_bars"]) == [0, 1, 2, 3, 0, 0, 1]


def test_7_max_drawdown_amount_in_account_currency():
    df = _make_eq([2000.0, 2500.0, 2100.0, 2600.0])
    dd = compute_drawdown_series(df, initial_balance=2000.0)
    summary = summarize_drawdown(dd, net_trading_profit=600.0)
    assert math.isclose(summary["max_drawdown_amount"], 400.0, rel_tol=1e-9)


def test_8_max_drawdown_pct_and_episode_averages():
    df = _make_eq([1000.0, 900.0, 1000.0, 800.0, 1000.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    summary = summarize_drawdown(dd, net_trading_profit=0.0)
    # Episode 1 max DD = -10%, Episode 2 max DD = -20% -> avg_episode_drawdown_pct = -15%
    assert math.isclose(summary["max_drawdown_pct"], -20.0, rel_tol=1e-9)
    assert math.isclose(summary["max_drawdown_depth_pct"], 20.0, rel_tol=1e-9)
    assert math.isclose(summary["avg_episode_drawdown_pct"], -15.0, rel_tol=1e-9)
    # All 5 bars drawdown_pct: [0, -10, 0, -20, 0] -> mean = -6.0%
    assert math.isclose(summary["avg_drawdown_pct"], -6.0, rel_tol=1e-9)


def test_9_ulcer_index_calculation():
    # Drawdown pcts: [0, -10, 0, -20, 0] -> squared: [0, 100, 0, 400, 0] -> mean = 100 -> sqrt = 10.0
    df = _make_eq([1000.0, 900.0, 1000.0, 800.0, 1000.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    summary = summarize_drawdown(dd)
    assert math.isclose(summary["ulcer_index"], 10.0, rel_tol=1e-9)


def test_10_recovery_factor_calculation():
    df = _make_eq([1000.0, 800.0, 1500.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    # Net profit = +500, Max DD amount = 200 -> Recovery Factor = 2.5
    summary = summarize_drawdown(dd, net_trading_profit=500.0)
    assert math.isclose(summary["recovery_factor"], 2.5, rel_tol=1e-9)


def test_11_performance_drawdown_and_account_drawdown_with_deposit_and_withdrawal():
    # Bar 0: 1000
    # Bar 1: -10% trading loss (-100 -> 900) + $500 deposit = 1400
    # Without cash-flow adjustment, naive equity (1400 > 1000) would think it is at a new peak (0% DD)!
    # With TWR Wealth Index and CF-adjusted HWM (1000 + 500 = 1500), Drawdown is -10% and -$100!
    eq_df = _make_eq([1000.0, 1400.0])
    ledger_df = pd.DataFrame(
        {
            "time": [eq_df.loc[1, "time"]],
            "transaction_type": [TransactionType.DEPOSIT.value],
            "amount": [500.0],
        }
    )
    dd = compute_drawdown_series(eq_df, ledger_df=ledger_df, initial_balance=1000.0)
    assert math.isclose(dd.loc[1, "drawdown_pct"], -10.0, rel_tol=1e-9)
    assert math.isclose(dd.loc[1, "account_peak_equity"], 1500.0, rel_tol=1e-9)
    assert math.isclose(dd.loc[1, "account_drawdown_depth"], 100.0, rel_tol=1e-9)


def test_12_strictly_increasing_series_has_zero_drawdown_and_no_episodes():
    df = _make_eq([1000.0, 1050.0, 1100.0, 1200.0])
    dd = compute_drawdown_series(df, initial_balance=1000.0)
    eps = extract_drawdown_episodes(dd)
    summary = summarize_drawdown(dd, net_trading_profit=200.0, episodes_df=eps)
    assert eps.empty
    assert list(eps.columns) == DRAWDOWN_EPISODE_COLUMNS
    assert summary["max_drawdown_pct"] == 0.0
    assert summary["max_drawdown_amount"] == 0.0
    assert summary["max_drawdown_duration_bars"] == 0
    assert math.isinf(summary["recovery_factor"])
