"""Unit tests cho H6 – Friction Stress Test & H7 – Parameter Stability."""
from __future__ import annotations

import copy
from typing import ClassVar
import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.validation.config import StabilityConfig
from quant_backtester.validation.parameter_stability import analyze_parameter_stability
from quant_backtester.validation.stress_test import (
    estimate_break_even_multiplier,
    run_friction_stress_test,
)


def _make_sine_market(n: int = 120) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    closes = 2000.0 + 15.0 * np.sin(t * 0.4)
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


class SyntheticPeakStrategy(Strategy):
    """Strategy giả lập giúp kiểm thử phân biệt vùng tham số mượt vs Isolated Peak."""

    name = "synthetic_peak"
    SMOOTH_MODE: ClassVar[bool] = True

    def __init__(self, p1: int = 0, p2: int = 0):
        super().__init__(p1=p1, p2=p2)
        self.p1 = p1
        self.p2 = p2

    @property
    def warmup_period(self) -> int:
        return 2

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        if self.p1 == 2 and self.p2 == 2:
            # Điểm đỉnh: giao dịch liên tục trên pha tăng của sóng sin
            df["entry_signal"] = np.where(df["close"].diff().fillna(0) > 0, 1, 0)
            df["exit_long_signal"] = df["entry_signal"] == 0
        elif self.SMOOTH_MODE:
            # Vùng lân cận mượt vẫn có lãi tương đương
            df["entry_signal"] = np.where(df["close"].diff().fillna(0) > 0.5, 1, 0)
            df["exit_long_signal"] = df["entry_signal"] == 0
        else:
            # Isolated Peak: các tham số lân cận giao dịch ngược sóng (lỗ nặng)
            df["entry_signal"] = np.where(df["close"].diff().fillna(0) > 0, -1, 0)
            df["exit_short_signal"] = df["entry_signal"] == 0
        return StrategySignals(df)


def test_friction_stress_test_scales_spread_commission_slippage_without_mutating_config():
    df = _make_sine_market(120)
    cfg = BacktestConfig(
        initial_balance=1000.0,
        spread_price=0.2,
        fee_rate_per_side=0.0005,
        commission_per_lot=5.0,
        slippage_price=0.05,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
    )
    cfg_orig = copy.deepcopy(cfg)

    report = run_friction_stress_test(
        df,
        MeanReversionStrategy(10, 1.4),
        cfg,
        multipliers=(0.0, 1.0, 2.0, 4.0),
        stress_types=("SPREAD", "COMMISSION", "SLIPPAGE", "COMBINED"),
    )

    # Config gốc tuyệt đối không đổi
    assert cfg.spread.spread_price == cfg_orig.spread.spread_price
    assert cfg.fee.commission_per_lot == cfg_orig.fee.commission_per_lot
    assert cfg.slippage.slippage_price == cfg_orig.slippage.slippage_price

    # Controlled Commission Test: giữ nguyên giao dịch, tổng phí tăng & net_pnl không tăng
    assert report.is_controlled_commission_monotonic
    ctrl = report.controlled_commission_df
    assert (np.diff(ctrl["total_fees"].to_numpy()) >= -1e-9).all()
    assert (np.diff(ctrl["net_pnl"].to_numpy()) <= 1e-9).all()

    # Full Engine scenarios đủ 4 loại x 4 hệ số = 16 dòng
    assert len(report.scenarios_df) == 16
    comm_rows = report.scenarios_df[report.scenarios_df["stress_type"] == "COMMISSION"].sort_values("multiplier")
    assert comm_rows["commission_per_lot"].tolist() == pytest.approx([0.0, 5.0, 10.0, 20.0])


def test_break_even_multiplier_only_returns_valid_unique_crossing():
    df_cross = pd.DataFrame(
        [
            {"stress_type": "COMBINED", "multiplier": 0.0, "net_trading_profit": 100.0},
            {"stress_type": "COMBINED", "multiplier": 1.0, "net_trading_profit": 50.0},
            {"stress_type": "COMBINED", "multiplier": 2.0, "net_trading_profit": -50.0},
        ]
    )
    assert estimate_break_even_multiplier(df_cross, "COMBINED") == pytest.approx(1.5)

    # Không có điểm đổi dấu -> None (không nội suy giả)
    df_no_cross = pd.DataFrame(
        [
            {"stress_type": "COMBINED", "multiplier": 0.0, "net_trading_profit": 100.0},
            {"stress_type": "COMBINED", "multiplier": 2.0, "net_trading_profit": 40.0},
        ]
    )
    assert estimate_break_even_multiplier(df_no_cross, "COMBINED") is None


def test_parameter_stability_distinguishes_smooth_plateau_vs_isolated_peak():
    df = _make_sine_market(80)
    cfg = BacktestConfig(initial_balance=1000.0, fee_rate_per_side=0.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    grid = {"p1": [1, 2, 3], "p2": [1, 2, 3]}

    SyntheticPeakStrategy.SMOOTH_MODE = True
    rep_smooth = analyze_parameter_stability(
        df,
        SyntheticPeakStrategy,
        cfg,
        param_grid=grid,
        objective_metric="total_return_pct",
    )
    assert rep_smooth.n_total_combos == 9
    assert rep_smooth.neighbors_count == 8  # Không tính chính (2,2) vào 8 điểm lân cận
    assert rep_smooth.surface_matrix is not None
    assert rep_smooth.surface_matrix.shape == (3, 3)
    assert rep_smooth.profitable_param_ratio == pytest.approx(1.0)
    assert not rep_smooth.is_isolated_peak
    assert rep_smooth.is_stable_peak

    SyntheticPeakStrategy.SMOOTH_MODE = False
    rep_spike = analyze_parameter_stability(
        df,
        SyntheticPeakStrategy,
        cfg,
        param_grid=grid,
        objective_metric="total_return_pct",
        stability_config=StabilityConfig(max_relative_gap=0.4, min_profitable_ratio=0.5),
    )
    assert rep_spike.best_params == {"p1": 2, "p2": 2}
    assert rep_spike.neighborhood_mean_score < 0.0
    assert rep_spike.profitable_param_ratio == pytest.approx(1.0 / 9.0)
    assert rep_spike.is_isolated_peak
    assert not rep_spike.is_stable_peak


def test_parameter_stability_handles_boundary_neighbors_and_zero_mean_cv():
    df = _make_sine_market(60)
    cfg = BacktestConfig(initial_balance=1000.0, fee_rate_per_side=0.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    # Điểm tốt nhất ở góc (1, 1) của lưới 2x2 chỉ có 3 điểm lân cận
    SyntheticPeakStrategy.SMOOTH_MODE = False
    rep = analyze_parameter_stability(
        df,
        SyntheticPeakStrategy,
        cfg,
        param_grid={"p1": [2, 3], "p2": [2, 3]},
        objective_metric="total_return_pct",
    )
    assert rep.neighbors_count == 3
