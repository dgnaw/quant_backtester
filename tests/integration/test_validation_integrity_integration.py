"""Integration tests cho Group H – Gate 04 Validation & Backtest Integrity."""
from __future__ import annotations

import copy
from typing import ClassVar
import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.validation.config import MarketRegimeConfig, ValidationConfig
from quant_backtester.validation.integrity_report import (
    assemble_integrity_report,
    run_integrity_suite,
)
from quant_backtester.validation.models import ValidationStatus


def _make_dataset(n: int = 160, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    closes = 2000.0 + 14.0 * np.sin(t * 0.32) + np.cumsum(rng.normal(0.0, 0.7, size=n))
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC"),
            "open": opens,
            "high": np.maximum(opens, closes) + 2.5,
            "low": np.minimum(opens, closes) - 2.5,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


class FuturePeekStrategy(Strategy):
    name = "future_peek_integration"
    mode: ClassVar[StrategyMode] = StrategyMode.VECTORIZED

    @property
    def warmup_period(self) -> int:
        return 3

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = np.where(df["close"].shift(-1) > df["close"], 1, -1)
        df["exit_long_signal"] = df["entry_signal"] == -1
        df["exit_short_signal"] = df["entry_signal"] == 1
        return StrategySignals(df)


class NonDetIntegrationStrategy(Strategy):
    name = "nondet_integration"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN
    _COUNTER: ClassVar[int] = 0

    def on_bar(self, bar: pd.Series, context) -> Signal | None:
        NonDetIntegrationStrategy._COUNTER += 1
        if NonDetIntegrationStrategy._COUNTER % 3 == 0 and context.position.is_flat:
            return Signal(time=pd.Timestamp(bar["time"]), symbol=context.symbol, signal_type=SignalType.LONG)
        if not context.position.is_flat and context.bar_index % 2 == 0:
            return Signal(time=pd.Timestamp(bar["time"]), symbol=context.symbol, signal_type=SignalType.EXIT_LONG)
        return None


def test_scenario_1_and_6_complete_validation_gate04_no_side_effects():
    t = np.arange(160, dtype=float)
    closes = 2000.0 + 25.0 * np.sin(t * 0.45)
    opens = np.r_[closes[0], closes[:-1]]
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=160, freq="1D", tz="UTC"),
            "open": opens,
            "high": np.maximum(opens, closes) + 2.5,
            "low": np.minimum(opens, closes) - 2.5,
            "close": closes,
            "volume": [100.0] * 160,
        }
    )
    df_orig = df.copy(deep=True)
    cfg = BacktestConfig(initial_balance=10_000.0, bars_per_year=252, spread_price=0.0, fee_rate_per_side=0.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    cfg_orig = copy.deepcopy(cfg)
    strat = MeanReversionStrategy(window=10, std_multiplier=1.1)
    base_res = BacktestEngine(cfg).run(df, strat)
    base_eq_orig = base_res.equity.copy(deep=True)
    base_led_orig = base_res.ledger.copy(deep=True)

    vcfg = ValidationConfig(
        random_seed=42,
        sample_step=40,
        max_lookahead_samples=3,
        walk_forward_folds=4,
        min_train_trades=0,
        strict_gate=True,
    )
    report = run_integrity_suite(
        df,
        strat,
        cfg,
        vcfg,
        baseline_result=base_res,
        param_grid={"window": [8, 10, 12], "std_multiplier": [1.0, 1.2]},
        stress_multipliers=(0.0, 1.0, 2.0),
        regime_config=MarketRegimeConfig(trend_window=15, vol_window=15),
    )

    assert set(report.statuses.keys()) == {
        "H1_LOOKAHEAD",
        "H2_WARMUP",
        "H3_IS_OOS",
        "H4_WALK_FORWARD",
        "H5_REPRODUCIBILITY",
        "H6_STRESS_TEST",
        "H7_PARAMETER_STABILITY",
        "H8_REGIME_ANALYSIS",
        "H9_OVERFITTING",
    }
    assert report.statuses["H1_LOOKAHEAD"] == ValidationStatus.PASS
    assert report.statuses["H2_WARMUP"] == ValidationStatus.PASS
    assert report.statuses["H5_REPRODUCIBILITY"] == ValidationStatus.PASS
    assert all(st != ValidationStatus.NOT_RUN for st in report.statuses.values())
    assert report.gate_04_passed

    # Không sửa đổi dữ liệu đầu vào hay kết quả gốc (DH1)
    pd.testing.assert_frame_equal(df, df_orig)
    pd.testing.assert_frame_equal(base_res.equity, base_eq_orig)
    pd.testing.assert_frame_equal(base_res.ledger, base_led_orig)
    assert cfg.fee.fee_rate_per_side == cfg_orig.fee.fee_rate_per_side
    assert strat.window == 10


def test_scenario_2_lookahead_strategy_fails_gate04():
    df = _make_dataset(45)
    cfg = BacktestConfig(initial_balance=10_000.0, bars_per_year=252)
    vcfg = ValidationConfig(sample_step=8, max_lookahead_samples=4, strict_gate=True)

    report = run_integrity_suite(
        df,
        FuturePeekStrategy(),
        cfg,
        vcfg,
        enable_walk_forward=False,
        enable_stress_test=False,
        enable_parameter_stability=False,
        enable_overfitting=False,
    )
    assert report.statuses["H1_LOOKAHEAD"] == ValidationStatus.FAIL
    assert report.statuses["H4_WALK_FORWARD"] == ValidationStatus.NOT_RUN
    assert not report.gate_04_passed
    assert report.overall_status == ValidationStatus.FAIL


def test_scenario_3_warmup_violation_fails_gate04():
    df = _make_dataset(30)
    cfg = BacktestConfig(initial_balance=10_000.0, bars_per_year=252)
    strat = MeanReversionStrategy(window=10, std_multiplier=1.4)
    res = BacktestEngine(cfg).run(df, strat, trade_start=10)
    # Giả lập Fill lọt vào vùng warm-up
    if not res.order_events.empty:
        res.order_events.loc[res.order_events["event_type"] == "FILLED", "bar_index"] = 2

    from quant_backtester.validation.warmup_guard import verify_warmup_boundary

    w_rep = verify_warmup_boundary(res, strategy=strat)
    gate = assemble_integrity_report(baseline_result=res, warmup_report=w_rep, strict_gate=True)
    assert gate.statuses["H2_WARMUP"] == ValidationStatus.FAIL
    assert not gate.gate_04_passed


def test_scenario_4_non_deterministic_strategy_fails_gate04():
    NonDetIntegrationStrategy._COUNTER = 0
    df = _make_dataset(30)
    cfg = BacktestConfig(initial_balance=10_000.0, bars_per_year=252)
    vcfg = ValidationConfig(sample_step=10, max_lookahead_samples=3, check_backtest_state=False, strict_gate=True)

    report = run_integrity_suite(
        df,
        NonDetIntegrationStrategy(),
        cfg,
        vcfg,
        enable_lookahead=False,
        enable_train_test=False,
        enable_walk_forward=False,
        enable_stress_test=False,
        enable_parameter_stability=False,
        enable_regime_analysis=False,
        enable_overfitting=False,
    )
    assert report.statuses["H5_REPRODUCIBILITY"] == ValidationStatus.FAIL
    assert not report.gate_04_passed


def test_scenario_5_strict_vs_non_strict_gate_for_optional_checks():
    from quant_backtester.validation.overfitting import OverfittingReport

    overfit_fail = OverfittingReport(
        status=ValidationStatus.FAIL,
        probabilistic_sharpe_ratio=0.4,
        deflated_sharpe_ratio=0.2,
        expected_max_sharpe_null=0.5,
        n_trials=20,
        raw_p_value=0.6,
        adjusted_p_value=1.0,
        is_oos_return_degradation=-0.2,
        is_oos_sharpe_degradation=-0.1,
        walk_forward_efficiency=-0.1,
        is_oos_rank_correlation=-0.8,
        triggered_flags=["LOW_DSR", "LOW_WFE"],
        warnings=["High overfitting risk."],
    )
    rep_strict = assemble_integrity_report(overfitting_report=overfit_fail, strict_gate=True)
    assert not rep_strict.gate_04_passed
    assert rep_strict.overall_status == ValidationStatus.FAIL

    # Với strict_gate=False và H1/H2/H5 không FAIL, FAIL ở mục không bắt buộc chỉ cảnh báo
    rep_lenient = assemble_integrity_report(overfitting_report=overfit_fail, strict_gate=False)
    assert rep_lenient.gate_04_passed
    assert rep_lenient.overall_status == ValidationStatus.WARNING
