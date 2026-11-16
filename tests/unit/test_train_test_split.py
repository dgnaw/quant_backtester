"""Unit tests cho H3 – Chronological IS/OOS Split & Evaluation (Giai đoạn 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.validation.train_test import (
    run_in_sample_out_of_sample,
    split_in_sample_out_of_sample,
)


def _make_ohlcv(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 2000.0 + np.cumsum(rng.normal(0.0, 3.0, size=n))
    o = np.r_[2000.0, c[:-1]]
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
            "open": o,
            "high": np.maximum(o, c) + 2.0,
            "low": np.minimum(o, c) - 2.0,
            "close": c,
            "volume": [100.0] * n,
        }
    )


class AlwaysLongForceCloseProbe(Strategy):
    name = "always_long_probe"

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


def test_split_by_ratio_time_and_index_chronological():
    df = _make_ohlcv(100)

    s_ratio = split_in_sample_out_of_sample(df, train_ratio=0.7, warmup_bars=10)
    assert len(s_ratio.in_sample_data) == 70
    assert len(s_ratio.oos_raw_data) == 30
    assert s_ratio.oos_trade_start == 10
    assert len(s_ratio.out_of_sample_data) == 40
    assert s_ratio.is_end_time < s_ratio.oos_start_time

    split_ts = df["time"].iloc[60]
    s_time = split_in_sample_out_of_sample(df, split_time=split_ts, warmup_bars=15)
    assert len(s_time.in_sample_data) == 60
    assert s_time.oos_start_time == split_ts
    assert s_time.oos_trade_start == 15

    s_idx = split_in_sample_out_of_sample(df, split_index=50, warmup_bars=5)
    assert len(s_idx.in_sample_data) == 50
    assert len(s_idx.oos_raw_data) == 50
    assert s_idx.oos_trade_start == 5


def test_embargo_creates_exact_gap_and_prefix_before_oos():
    df = _make_ohlcv(100)
    s = split_in_sample_out_of_sample(df, split_index=60, embargo_bars=5, warmup_bars=10)
    assert len(s.in_sample_data) == 60
    assert s.is_end_time == df["time"].iloc[59]
    assert s.oos_start_time == df["time"].iloc[65]
    assert len(s.oos_raw_data) == 35
    # Warm-up prefix lấy 10 nến ngay trước index 65 (tức [55..64])
    assert s.oos_trade_start == 10
    assert s.out_of_sample_data["time"].iloc[0] == df["time"].iloc[55]


def test_run_in_sample_out_of_sample_force_closes_is_and_isolates_oos_metrics():
    df = _make_ohlcv(100)
    cfg = BacktestConfig(initial_balance=1000.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    strat = AlwaysLongForceCloseProbe(warmup=5)

    ev = run_in_sample_out_of_sample(df, strat, cfg, split_index=60, embargo_bars=0)
    # Vị thế mở cuối IS phải bị force-close tại IS
    assert ev.is_force_close_summary.closed_positions_count == 1
    assert len(ev.in_sample_result.open_positions) == 0
    # Mọi lệnh mở trong OOS chỉ được vào từ oos_start_time trở đi
    if not ev.out_of_sample_result.trades.empty:
        assert pd.to_datetime(ev.out_of_sample_result.trades["entry_time"]).min() >= ev.split.oos_start_time
    # OOS metrics tính đúng trên số nến OOS thực tế (40), KHÔNG gộp warm-up prefix
    assert ev.oos_window.evaluation_bars == 40
    assert ev.oos_window.return_observations == 40
    assert ev.out_of_sample_metrics["evaluation_bars"] == 40


def test_is_oos_with_mean_reversion_and_validation_errors():
    df = _make_ohlcv(120)
    cfg = BacktestConfig(initial_balance=1000.0)
    ev = run_in_sample_out_of_sample(df, MeanReversionStrategy(15, 1.5), cfg, train_ratio=0.7)
    assert "sharpe_degradation_ratio" in ev.degradation_summary

    with pytest.raises(ValueError, match="train_ratio"):
        split_in_sample_out_of_sample(df, train_ratio=1.2)
    with pytest.raises(ValueError, match="rỗng"):
        split_in_sample_out_of_sample(pd.DataFrame())
    bad_ts = df.copy()
    bad_ts.loc[10, "time"] = bad_ts.loc[5, "time"]
    with pytest.raises(ValueError, match="Timestamp"):
        split_in_sample_out_of_sample(bad_ts)
    with pytest.raises(ValueError, match="warmup_period"):
        run_in_sample_out_of_sample(df.iloc[:15], MeanReversionStrategy(20, 1.5), cfg, train_ratio=0.7)
