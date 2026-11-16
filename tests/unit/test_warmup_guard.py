"""Unit tests cho H2 – Warm-up Boundary & Evaluation Window Isolation (Giai đoạn 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.validation.models import ValidationStatus
from quant_backtester.validation.warmup_guard import (
    compute_evaluation_metrics,
    extract_evaluation_window,
    resolve_boundaries,
    verify_warmup_boundary,
)


class AlwaysLongStrategy(Strategy):
    name = "always_long_test"

    def __init__(self, warmup: int = 5):
        super().__init__(warmup=warmup)
        self._w = warmup

    @property
    def warmup_period(self) -> int:
        return self._w

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = 1
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)


def _make_trending_bars(n: int = 20) -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    closes = [100.0 + i * 2.0 for i in range(n)]
    opens = [100.0] + closes[:-1]
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": [max(o, c) + 1.0 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 1.0 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def test_boundary_resolution_separates_warmup_signal_and_trade_start():
    assert resolve_boundaries(trade_start_index=10, strict_signal_start=False) == (0, 9, 10)
    assert resolve_boundaries(trade_start_index=10, strict_signal_start=True) == (0, 10, 10)
    assert resolve_boundaries(trade_start_index=0, strict_signal_start=False) == (0, 0, 0)


def test_valid_run_passes_warmup_boundary_audit():
    df = _make_trending_bars(20)
    strat = AlwaysLongStrategy(warmup=6)
    cfg = BacktestConfig(initial_balance=1000.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    res = BacktestEngine(cfg).run(df, strat, trade_start=8)

    report = verify_warmup_boundary(res, strategy=strat)
    assert report.status == ValidationStatus.PASS
    assert report.is_valid
    assert report.trade_start_index == 8
    assert report.signal_start_index == 7
    assert report.first_fill_bar_index == 8


def test_detects_fill_trade_and_equity_mutation_inside_warmup():
    df = _make_trending_bars(15)
    strat = AlwaysLongStrategy(warmup=5)
    cfg = BacktestConfig(initial_balance=1000.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    res = BacktestEngine(cfg).run(df, strat, trade_start=5)

    # Giả lập sai lệch trong vùng warm-up
    res.equity.loc[2, "equity"] = 1050.0
    res.equity.loc[2, "position"] = 1
    res.trades.loc[0, "entry_bar_index"] = 3
    res.order_events.loc[res.order_events["event_type"] == "FILLED", "bar_index"] = 3

    report = verify_warmup_boundary(res, strategy=strat)
    assert report.status == ValidationStatus.FAIL
    assert not report.is_valid
    joined = " | ".join(report.violations)
    assert "EQUITY_CHANGED_IN_WARMUP" in joined
    assert "POSITION_IN_WARMUP" in joined
    assert "TRADE_ENTRY_IN_WARMUP" in joined
    assert "FILL_IN_WARMUP" in joined


def test_extract_evaluation_window_excludes_warmup_zeros_and_preserves_anchor():
    df = _make_trending_bars(20)
    strat = AlwaysLongStrategy(warmup=10)
    cfg = BacktestConfig(initial_balance=1000.0, lot_size=0.01, contract_size=100.0, fee_rate_per_side=0.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    res = BacktestEngine(cfg).run(df, strat, trade_start=10)

    win = extract_evaluation_window(res)
    assert win.evaluation_start_index == 10
    assert win.evaluation_bars == 10  # Bars 10..19
    assert win.warmup_bars == 10
    assert win.excluded_bars == 10
    assert win.return_observations == 10
    assert win.anchor_equity == pytest.approx(1000.0)
    assert not win.synthetic_anchor

    metrics, _ = compute_evaluation_metrics(res, win)
    assert metrics["evaluation_bars"] == 10
    assert metrics["evaluation_return_observations"] == 10
    assert metrics["total_return_pct"] == pytest.approx(res.metrics["total_return_pct"])


def test_extract_evaluation_window_preserves_custom_anchor_and_cash_flow():
    df = _make_trending_bars(6)
    strat = AlwaysLongStrategy(warmup=0)
    cfg = BacktestConfig(initial_balance=1000.0, fee_rate_per_side=0.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    res = BacktestEngine(cfg).run(df, strat, trade_start=0)
    res.equity["equity"] = [1000.0, 1020.0, 1050.0, 1200.0, 1210.0, 1230.0]

    # Thêm cash flow giả ở bar 4 (thuộc evaluation window [3..6))
    res.ledger = pd.DataFrame(
        [
            {
                "entry_id": "LED-CF",
                "time": res.equity["time"].iloc[4],
                "timestamp": res.equity["time"].iloc[4],
                "transaction_type": "DEPOSIT",
                "amount": 100.0,
                "balance_after": 1150.0,
                "symbol": None,
                "order_id": None,
                "fill_id": None,
                "trade_id": None,
                "reason": "TOP_UP",
                "description": "TOP_UP",
            }
        ]
    )

    win = extract_evaluation_window(res, start_index=3)
    assert win.anchor_equity == pytest.approx(1050.0)
    assert win.evaluation_bars == 3
    assert len(win.evaluation_ledger_df) == 1


def test_extract_evaluation_window_single_bar_and_empty_equity_edge_cases():
    df = _make_trending_bars(1)
    cfg = BacktestConfig(initial_balance=1000.0)
    res = BacktestEngine(cfg).run(df, AlwaysLongStrategy(warmup=0))

    win = extract_evaluation_window(res)
    assert win.evaluation_bars == 1
    assert win.return_observations == 1
    assert win.synthetic_anchor

    res.equity = pd.DataFrame()
    empty_audit = verify_warmup_boundary(res)
    assert empty_audit.status == ValidationStatus.NOT_RUN
    with pytest.raises(ValueError, match="rỗng"):
        extract_evaluation_window(res)
