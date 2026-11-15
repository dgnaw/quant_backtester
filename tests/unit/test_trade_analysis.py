"""
Unit tests for Tasks G6 & G7: Initial Risk Tracking, Risk Allocation, R-Multiples,
SQN & Net-PnL Trade Analysis.
"""

from __future__ import annotations

import math
import pandas as pd

from quant_backtester.analytics.trade_analysis import analyze_trades
from quant_backtester.config import InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.portfolio import Portfolio


def _make_portfolio(mode: PositionMode = PositionMode.NETTING_SINGLE) -> Portfolio:
    acc = Account(initial_balance=10000.0)
    inst = InstrumentConfig(symbol="XAUUSD", contract_size=100.0, lot_size=0.10)
    return Portfolio(account=acc, instrument_config=inst, default_position_mode=mode)


def test_1_net_pnl_used_to_classify_win_loss_breakeven():
    # Trade 1: gross_pnl = +4.0, total_fee = 6.0 -> net_pnl = -2.0 (Must be classified as LOSS, not WIN!)
    # Trade 2: gross_pnl = +20.0, total_fee = 20.0 -> net_pnl = 0.0 (Breakeven)
    # Trade 3: gross_pnl = +50.0, total_fee = 5.0 -> net_pnl = +45.0 (Win)
    df = pd.DataFrame(
        {
            "gross_pnl": [4.0, 20.0, 50.0],
            "total_fee": [6.0, 20.0, 5.0],
            "net_pnl": [-2.0, 0.0, 45.0],
        }
    )
    stats = analyze_trades(df)
    assert stats["total_trades"] == 3
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["breakevens"] == 1
    assert math.isclose(stats["win_rate_pct"], 100.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(stats["loss_rate_pct"], 100.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(stats["breakeven_rate_pct"], 100.0 / 3.0, rel_tol=1e-9)


def test_2_gross_vs_net_pnl_and_cost_drag():
    df = pd.DataFrame(
        {
            "gross_pnl": [100.0, -40.0, 40.0],
            "total_fee": [10.0, 10.0, 10.0],
            "net_pnl": [90.0, -50.0, 30.0],
        }
    )
    stats = analyze_trades(df)
    # Gross sum = 100, Net sum = 70, Fees = 30 -> Cost Drag = 30 (30% of gross profit)
    assert math.isclose(stats["total_gross_pnl"], 100.0)
    assert math.isclose(stats["total_net_pnl"], 70.0)
    assert math.isclose(stats["total_fees"], 30.0)
    assert math.isclose(stats["cost_drag"], 30.0)
    assert math.isclose(stats["cost_drag_pct"] or 0.0, 30.0)
    assert math.isclose(stats["gross_profit_factor"], 140.0 / 40.0)
    assert math.isclose(stats["profit_factor"], 120.0 / 50.0)


def test_3_profit_factor_normal_and_edge_cases():
    # Only wins -> inf
    s_win = analyze_trades(pd.DataFrame({"net_pnl": [10.0, 20.0]}))
    assert math.isinf(s_win["profit_factor"])

    # Only losses -> 0.0
    s_loss = analyze_trades(pd.DataFrame({"net_pnl": [-10.0, -20.0]}))
    assert s_loss["profit_factor"] == 0.0

    # Only breakevens -> 0.0
    s_be = analyze_trades(pd.DataFrame({"net_pnl": [0.0, 0.0]}))
    assert s_be["profit_factor"] == 0.0


def test_4_expectancy_cash_per_trade():
    df = pd.DataFrame({"net_pnl": [100.0, -40.0, 60.0, -20.0]})
    stats = analyze_trades(df)
    assert math.isclose(stats["expectancy_cash_per_trade"], 25.0)


def test_5_avg_win_avg_loss_and_payoff_ratio():
    df = pd.DataFrame({"net_pnl": [100.0, 60.0, -40.0, -20.0]})
    stats = analyze_trades(df)
    # avg_win = 80.0, avg_loss = -30.0 -> payoff_ratio = 80 / 30 = 2.666667
    assert math.isclose(stats["avg_win"], 80.0)
    assert math.isclose(stats["avg_loss"], -30.0)
    assert math.isclose(stats["payoff_ratio"], 80.0 / 30.0)


def test_6_entry_initial_risk_recorded_in_entry_contribution():
    pf = _make_portfolio()
    # BUY 0.10 lots (10 units) @ 2000, SL = 1990, fee = 2.50
    # Default initial risk = (2000 - 1990)*10 + 2.50 + 2.50 = 105.0
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-01 09:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.10, 10.0, fee=2.50),
        stop_loss_price=1990.0,
    )
    pos = pf.get_position("XAUUSD")
    assert len(pos.contributions) == 1
    contrib = pos.contributions[0]
    assert contrib.initial_stop_loss_price == 1990.0
    assert math.isclose(contrib.initial_risk_amount or 0.0, 105.0)
    assert math.isclose(contrib.remaining_risk_amount or 0.0, 105.0)


def test_7_scale_in_risk_allocation_across_multiple_entries():
    pf = _make_portfolio(PositionMode.NETTING_SCALE_IN)
    # Entry 1: BUY 0.10 lots (10 units) @ 2000, SL = 1990, fee = 2.0 -> risk_1 = 100 + 4 = 104.0
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-01 09:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.10, 10.0, fee=2.0),
        stop_loss_price=1990.0,
    )
    # Entry 2: BUY 0.10 lots (10 units) @ 2005, SL = 1990, fee = 2.0 -> risk_2 = 150 + 4 = 154.0
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2025-01-01 10:00"), "XAUUSD", OrderSide.BUY, 2005.0, 0.10, 10.0, fee=2.0),
        stop_loss_price=1990.0,
    )
    pos = pf.get_position("XAUUSD")
    assert math.isclose(pos.remaining_risk_amount or 0.0, 258.0)

    # Full close 0.20 lots (20 units) @ 2020, fee = 4.0
    pf.on_fill(
        Fill("F-3", "O-3", pd.Timestamp("2025-01-01 11:00"), "XAUUSD", OrderSide.SELL, 2020.0, 0.20, 20.0, fee=4.0),
    )
    tr = pf.trade_records[0]
    assert math.isclose(tr.initial_risk_amount or 0.0, 258.0)
    assert math.isclose(tr.r_multiple or 0.0, tr.net_pnl / 258.0)
    assert pf.reconcile().is_valid


def test_8_partial_close_risk_allocation_conservation():
    pf = _make_portfolio(PositionMode.NETTING_SCALE_IN)
    # Entry: BUY 0.20 lots (20 units) @ 2000, SL = 1990, fee = 4.0 -> initial_risk = 200 + 8 = 208.0
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-01 09:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.20, 20.0, fee=4.0),
        stop_loss_price=1990.0,
    )
    # Partial close 25% (0.05 lots = 5 units) @ 2010, fee = 1.0 -> allocated_risk = 52.0, remaining_risk = 156.0
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2025-01-01 10:00"), "XAUUSD", OrderSide.SELL, 2010.0, 0.05, 5.0, fee=1.0),
    )
    pos = pf.get_position("XAUUSD")
    assert math.isclose(pos.remaining_risk_amount or 0.0, 156.0)
    assert math.isclose(pf.trade_records[0].initial_risk_amount or 0.0, 52.0)

    # Close remaining 75% (0.15 lots = 15 units) @ 2020, fee = 3.0 -> allocated_risk = 156.0
    pf.on_fill(
        Fill("F-3", "O-3", pd.Timestamp("2025-01-01 11:00"), "XAUUSD", OrderSide.SELL, 2020.0, 0.15, 15.0, fee=3.0),
    )
    assert math.isclose(pf.trade_records[1].initial_risk_amount or 0.0, 156.0)
    assert math.isclose(
        (pf.trade_records[0].initial_risk_amount or 0.0)
        + (pf.trade_records[1].initial_risk_amount or 0.0),
        208.0,
    )
    assert pf.reconcile().is_valid


def test_9_r_multiple_average_r_and_expectancy_r():
    pf = _make_portfolio()
    # Trade 1: +2R (initial_risk = 100.0, net_pnl = +200.0)
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-01 09:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.10, 10.0, fee=0.0),
        stop_loss_price=1990.0,
    )
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2025-01-01 10:00"), "XAUUSD", OrderSide.SELL, 2020.0, 0.10, 10.0, fee=0.0),
    )
    # Trade 2: -1R (initial_risk = 100.0, net_pnl = -100.0)
    pf.on_fill(
        Fill("F-3", "O-3", pd.Timestamp("2025-01-01 11:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.10, 10.0, fee=0.0),
        stop_loss_price=1990.0,
    )
    pf.on_fill(
        Fill("F-4", "O-4", pd.Timestamp("2025-01-01 12:00"), "XAUUSD", OrderSide.SELL, 1990.0, 0.10, 10.0, fee=0.0),
    )

    stats = analyze_trades(pf.trades_to_dataframe())
    assert stats["valid_r_trades"] == 2
    assert math.isclose(stats["r_coverage_pct"], 100.0)
    assert math.isclose(stats["average_r"] or 0.0, 0.5)
    assert math.isclose(stats["expectancy_r"] or 0.0, 0.5)
    assert math.isclose(stats["r_win_avg"] or 0.0, 2.0)
    assert math.isclose(stats["r_loss_avg"] or 0.0, -1.0)


def test_10_trades_missing_stop_loss_do_not_fabricate_r_multiple():
    pf = _make_portfolio()
    # Trade without stop_loss_price -> r_multiple must be None
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2025-01-01 09:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.10, 10.0, fee=1.0),
        stop_loss_price=None,
    )
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2025-01-01 10:00"), "XAUUSD", OrderSide.SELL, 1990.0, 0.10, 10.0, fee=1.0),
    )
    tr = pf.trade_records[0]
    assert tr.initial_risk_amount is None
    assert tr.r_multiple is None

    stats = analyze_trades(pf.trades_to_dataframe())
    assert stats["valid_r_trades"] == 0
    assert stats["r_coverage_pct"] == 0.0
    assert stats["average_r"] is None
    assert stats["expectancy_r"] is None
    assert stats["sqn"] is None
    # Empirical R is provided separately
    assert math.isclose(stats["empirical_r_multiple"] or 0.0, -1.0)


def test_11_sqn_computed_on_r_multiples_vs_sqn_cash_proxy():
    df = pd.DataFrame(
        {
            "net_pnl": [200.0, -100.0, 300.0, -100.0],
            "r_multiple": [2.0, -1.0, 3.0, -1.0],
        }
    )
    stats = analyze_trades(df)
    r_arr = [2.0, -1.0, 3.0, -1.0]
    r_mean = sum(r_arr) / 4.0
    r_std = float(pd.Series(r_arr).std(ddof=1))
    expected_sqn = (r_mean / r_std) * math.sqrt(4.0)
    assert math.isclose(stats["sqn"] or 0.0, expected_sqn, rel_tol=1e-9)
    assert stats["sqn_cash_proxy"] is not None


def test_12_consecutive_win_and_loss_streaks_with_breakeven_breaker():
    # Sequence: W, W, W, 0 (breaks streak!), W, W, L, L, L, L, W
    df = pd.DataFrame(
        {
            "net_pnl": [10.0, 20.0, 15.0, 0.0, 12.0, 18.0, -5.0, -8.0, -12.0, -4.0, 25.0],
        }
    )
    stats = analyze_trades(df)
    assert stats["max_consecutive_wins"] == 3
    assert stats["max_consecutive_losses"] == 4


def test_13_long_and_short_breakdown():
    df = pd.DataFrame(
        {
            "direction": ["LONG", "LONG", "SHORT", "SHORT", "SHORT"],
            "net_pnl": [50.0, -20.0, 30.0, 40.0, -10.0],
        }
    )
    stats = analyze_trades(df)
    assert stats["long_trades"] == 2
    assert stats["short_trades"] == 3
    assert math.isclose(stats["long_win_rate_pct"], 50.0)
    assert math.isclose(stats["short_win_rate_pct"], 200.0 / 3.0)
    assert math.isclose(stats["long_net_pnl"], 30.0)
    assert math.isclose(stats["short_net_pnl"], 60.0)


def test_14_holding_bars_winner_and_loser_averages():
    df = pd.DataFrame(
        {
            "net_pnl": [50.0, 30.0, -20.0],
            "holding_bars": [4, 8, 3],
            "weighted_holding_bars": [4.0, 8.0, 3.0],
        }
    )
    stats = analyze_trades(df)
    assert math.isclose(stats["avg_holding_bars"] or 0.0, 5.0)
    assert math.isclose(stats["avg_winner_holding_bars"] or 0.0, 6.0)
    assert math.isclose(stats["avg_loser_holding_bars"] or 0.0, 3.0)


def test_15_slippage_cost_is_diagnostic_and_not_subtracted_twice():
    pf = _make_portfolio()
    # Entry @ 2000.50 (includes 0.50 slippage = $5.0 slippage_cost), fee = $2.0
    pf.on_fill(
        Fill(
            "F-1", "O-1", pd.Timestamp("2025-01-01 09:00"), "XAUUSD", OrderSide.BUY,
            2000.50, 0.10, 10.0, fee=2.0, slippage=0.50, metadata={"slippage_cost": 5.0}
        )
    )
    # Exit @ 2010.50 (includes 0.50 slippage = $5.0 slippage_cost), fee = $2.0
    pf.on_fill(
        Fill(
            "F-2", "O-2", pd.Timestamp("2025-01-01 10:00"), "XAUUSD", OrderSide.SELL,
            2010.50, 0.10, 10.0, fee=2.0, slippage=0.50, metadata={"slippage_cost": 5.0}
        )
    )
    stats = analyze_trades(pf.trades_to_dataframe())
    # Gross PnL = (2010.50 - 2000.50)*10 = 100.0
    # Net PnL = 100.0 - 4.0 = 96.0 (NOT 96.0 - 10.0 = 86.0!)
    assert math.isclose(stats["total_gross_pnl"], 100.0)
    assert math.isclose(stats["total_net_pnl"], 96.0)
    assert math.isclose(stats["total_fees"], 4.0)
    assert math.isclose(stats["total_slippage_cost"], 10.0)


def test_16_zero_trades_handling():
    stats = analyze_trades(pd.DataFrame())
    assert stats["total_trades"] == 0
    assert stats["wins"] == 0
    assert stats["losses"] == 0
    assert stats["profit_factor"] == 0.0
    assert stats["average_r"] is None
    assert stats["sqn"] is None
