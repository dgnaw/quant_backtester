"""Unit tests cho Grid Runner (DH5) & Walk-Forward Analysis (H4)."""
from __future__ import annotations

from typing import ClassVar
import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.strategies.parameters import Parameter
from quant_backtester.walk_forward import walk_forward_mean_reversion
from quant_backtester.validation.grid_runner import (
    GridRunResult,
    build_param_grid,
    instantiate_fresh_strategy,
    run_parameter_grid,
    select_best_run,
)
from quant_backtester.validation.walk_forward import WalkForwardMode, run_walk_forward


def _make_oscillating_bars(n: int = 300, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    closes = 2000.0 + 18.0 * np.sin(t * 0.35) + np.cumsum(rng.normal(0.0, 0.8, size=n))
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
            "open": opens,
            "high": np.maximum(opens, closes) + 2.0,
            "low": np.minimum(opens, closes) - 2.0,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


class DualSmaCrossStrategy(Strategy):
    """Strategy khác (không phải MeanReversion) để chứng minh Walk-Forward Strategy-Agnostic."""

    name = "dual_sma_cross"
    parameters: ClassVar[tuple[Parameter, ...]] = (
        Parameter(name="fast", param_type=int, default=5, min_value=3, max_value=9, step=3, optimizable=True),
        Parameter(name="slow", param_type=int, default=15, min_value=12, max_value=18, step=3, optimizable=True),
    )

    def __init__(self, fast: int = 5, slow: int = 15, **kwargs):
        super().__init__(fast=fast, slow=slow, **kwargs)
        self._touched = 0

    @property
    def warmup_period(self) -> int:
        return int(self.slow)

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        self._touched += 1
        df = market_data.copy()
        sma_f = df["close"].rolling(self.fast, min_periods=self.fast).mean()
        sma_s = df["close"].rolling(self.slow, min_periods=self.slow).mean()
        df["entry_signal"] = 0
        df.loc[sma_f > sma_s, "entry_signal"] = 1
        df.loc[sma_f < sma_s, "entry_signal"] = -1
        df["exit_long_signal"] = sma_f < sma_s
        df["exit_short_signal"] = sma_f > sma_s
        return StrategySignals(df)


def test_grid_builder_and_fresh_strategy_instance_isolation():
    combos = build_param_grid(DualSmaCrossStrategy)
    assert len(combos) == 9  # fast in [3,6,9] x slow in [12,15,18]

    orig = DualSmaCrossStrategy(fast=3, slow=12)
    f1 = instantiate_fresh_strategy(orig, {"fast": 6})
    f2 = instantiate_fresh_strategy(orig, {"fast": 6})
    assert f1 is not orig
    assert f1 is not f2
    assert f1.fast == 6 and f1.slow == 12
    assert orig.fast == 3


def test_select_best_run_tie_breaks_by_grid_order_and_ignores_invalid():
    runs = [
        GridRunResult(run_index=0, params={"a": 1}, score=None, is_valid=False),
        GridRunResult(run_index=1, params={"a": 2}, score=1.5, is_valid=True),
        GridRunResult(run_index=2, params={"a": 3}, score=1.5, is_valid=True),
        GridRunResult(run_index=3, params={"a": 4}, score=0.8, is_valid=True),
    ]
    best = select_best_run(runs)
    assert best is not None
    assert best.run_index == 1  # Điểm bằng nhau -> giữ run xuất hiện trước


def test_rolling_and_anchored_walk_forward_windows_and_chronological_oos_stitching():
    df = _make_oscillating_bars(240)
    cfg = BacktestConfig(initial_balance=1000.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    grid = {"fast": [4, 6], "slow": [12, 16]}

    wf_roll = run_walk_forward(
        df,
        DualSmaCrossStrategy,
        cfg,
        param_grid=grid,
        mode=WalkForwardMode.ROLLING,
        train_bars=90,
        test_bars=40,
        embargo_bars=0,
        objective_metric="total_return_pct",
    )
    assert len(wf_roll.folds) == 3
    for idx, f in enumerate(wf_roll.folds):
        assert f.train_bars == 90
        assert f.test_bars == 40
        if idx > 0:
            # OOS liên tục khi embargo_bars == 0, không overlap
            assert f.test_start_index == wf_roll.folds[idx - 1].test_end_index
            assert f.test_start_time > wf_roll.folds[idx - 1].test_end_time

    assert len(wf_roll.stitched_oos_equity) == 120
    assert wf_roll.stitched_oos_equity["fold_id"].tolist() == [1] * 40 + [2] * 40 + [3] * 40
    # Kiểm tra Wealth Index ghép nối chính xác từ TWR từng bar
    recalc_eq = 1000.0 * np.cumprod(1.0 + wf_roll.stitched_oos_equity["twr_return"].to_numpy())
    assert np.allclose(wf_roll.stitched_oos_equity["equity"].to_numpy(), recalc_eq, rtol=1e-9)

    # ANCHORED mở rộng IS từ bar 0
    wf_anch = run_walk_forward(
        df,
        DualSmaCrossStrategy,
        cfg,
        param_grid=grid,
        mode=WalkForwardMode.ANCHORED,
        train_bars=90,
        test_bars=40,
        embargo_bars=5,
        objective_metric="total_return_pct",
    )
    assert len(wf_anch.folds) == 3
    assert [f.train_bars for f in wf_anch.folds] == [90, 130, 170]
    assert [f.embargo_bars for f in wf_anch.folds] == [5, 5, 5]
    assert [f.oos_gap_bars for f in wf_anch.folds] == [0, 0, 0]


def test_min_train_trades_filtering_and_wfe_non_positive_is_handling():
    df = _make_oscillating_bars(150)
    cfg = BacktestConfig(initial_balance=1000.0)
    wf_high_min = run_walk_forward(
        df,
        DualSmaCrossStrategy,
        cfg,
        param_grid={"fast": [5], "slow": [15]},
        folds=3,
        min_train_trades=999,
    )
    assert len(wf_high_min.folds) == 0
    assert any("SKIPPED" in m for m in wf_high_min.messages)

    # Kiểm tra trường hợp IS lỗ -> WFE phải trả về None (không chia hai số âm)
    from quant_backtester.validation.walk_forward import _compound_is_annualized_return

    wf = run_walk_forward(
        df,
        MeanReversionStrategy,
        cfg,
        param_grid={"window": [10, 15], "std_multiplier": [1.5]},
        folds=3,
        objective_metric="total_return_pct",
    )
    if wf.is_annualized_return_pct is not None and wf.is_annualized_return_pct <= 0.0:
        assert wf.walk_forward_efficiency is None
    else:
        assert wf.walk_forward_efficiency is not None or "WFE_UNDEFINED" in " ".join(wf.messages)


def test_legacy_walk_forward_mean_reversion_wrapper_preserves_schema():
    df = _make_oscillating_bars(200)
    cfg = BacktestConfig(initial_balance=1000.0)
    folds_df, summary_df = walk_forward_mean_reversion(
        df, cfg, windows=(15, 20), multipliers=(1.5, 2.0), folds=4
    )
    for col in ("fold", "window", "std_multiplier", "train_sharpe", "test_return_pct", "test_max_drawdown_pct", "test_trades"):
        assert col in folds_df.columns
    for col in ("combined_oos_return_pct", "combined_oos_max_drawdown_pct", "combined_oos_sharpe", "oos_bars"):
        assert col in summary_df.columns
    assert int(summary_df["oos_bars"].iloc[0]) == int(folds_df["test_bars"].sum())
