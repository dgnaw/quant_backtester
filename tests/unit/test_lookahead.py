"""Unit tests cho H1 – Anti-Lookahead Verification (Giai đoạn 1)."""
from __future__ import annotations

from typing import ClassVar
import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.order import OrderOrigin
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.validation.config import ValidationConfig
from quant_backtester.validation.lookahead import (
    audit_order_eligibility,
    verify_no_lookahead,
)
from quant_backtester.validation.models import ValidationStatus


def _make_market(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, size=n)
    closes = 2000.0 * np.exp(np.cumsum(rets))
    opens = np.r_[2000.0, closes[:-1]]
    highs = np.maximum(opens, closes) + np.abs(rng.normal(1.5, 0.5, size=n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(1.5, 0.5, size=n))
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 100.0),
        }
    )


class ShiftMinusOneLookaheadStrategy(Strategy):
    """Strategy cố tình nhìn trước 1 nến bằng shift(-1)."""

    name = "shift_minus_one_lookahead"
    mode: ClassVar[StrategyMode] = StrategyMode.VECTORIZED

    @property
    def warmup_period(self) -> int:
        return 2

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        future_close = df["close"].shift(-1)
        df["entry_signal"] = np.where(future_close > df["close"] * 1.001, 1, 0)
        df["exit_long_signal"] = df["entry_signal"] == 0
        df["exit_short_signal"] = False
        return StrategySignals(df)


class GlobalMeanLookaheadStrategy(Strategy):
    """Strategy cố tình dùng trung bình toàn bộ DataFrame (global leak)."""

    name = "global_mean_lookahead"
    mode: ClassVar[StrategyMode] = StrategyMode.VECTORIZED

    @property
    def warmup_period(self) -> int:
        return 2

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        global_avg = float(df["close"].mean())
        df["entry_signal"] = np.where(df["close"] < global_avg, 1, -1)
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)


class CausalEventStrategy(Strategy):
    """Strategy EVENT_DRIVEN hợp lệ chỉ dùng context.history()."""

    name = "causal_event_strategy"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    @property
    def warmup_period(self) -> int:
        return 5

    def on_bar(self, bar: pd.Series, context) -> Signal | None:
        hist = context.history(5)
        if len(hist) < 5:
            return None
        sma = float(hist["close"].mean())
        ts = pd.Timestamp(bar["time"])
        if context.position.is_flat and float(bar["close"]) < sma:
            return Signal(time=ts, symbol=context.symbol, signal_type=SignalType.LONG)
        if not context.position.is_flat and float(bar["close"]) > sma:
            return Signal(time=ts, symbol=context.symbol, signal_type=SignalType.EXIT_LONG)
        return None


def test_valid_vectorized_strategy_passes_lookahead_and_preserves_inputs():
    df = _make_market(60)
    df_copy = df.copy(deep=True)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0)
    strat = MeanReversionStrategy(window=10, std_multiplier=1.5)

    report = verify_no_lookahead(df, strat, config=cfg, sample_step=10)
    assert report.status == ValidationStatus.PASS
    assert report.strategy_status == ValidationStatus.PASS
    assert report.engine_status == ValidationStatus.PASS
    assert report.is_valid
    assert len(report.violations) == 0
    pd.testing.assert_frame_equal(df, df_copy)


def test_valid_event_driven_strategy_passes_lookahead():
    df = _make_market(40)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0)
    strat = CausalEventStrategy()

    report = verify_no_lookahead(df, strat, config=cfg, sample_step=8, run_mode=StrategyMode.EVENT_DRIVEN)
    assert report.status == ValidationStatus.PASS
    assert report.strategy_status == ValidationStatus.PASS
    assert report.engine_status == ValidationStatus.PASS


def test_shift_minus_one_strategy_detected_by_prefix_and_perturbation():
    df = _make_market(40)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0)

    report = verify_no_lookahead(df, ShiftMinusOneLookaheadStrategy(), config=cfg, sample_step=3)
    assert report.status == ValidationStatus.FAIL
    assert report.strategy_status == ValidationStatus.FAIL
    assert any(v.check_type == "PREFIX_SIGNAL" and v.source == "STRATEGY" for v in report.violations)
    assert any(v.check_type == "FUTURE_PERTURBATION" and v.source == "STRATEGY" for v in report.violations)


def test_global_dataframe_lookahead_strategy_detected():
    df = _make_market(35)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0)

    report = verify_no_lookahead(df, GlobalMeanLookaheadStrategy(), config=cfg, sample_step=5)
    assert report.status == ValidationStatus.FAIL
    assert report.strategy_status == ValidationStatus.FAIL
    assert len(report.strategy_violations) > 0


def test_order_eligibility_detects_same_bar_fill_engine_violation():
    df = _make_market(15)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0)
    res = BacktestEngine(cfg).run(df, MeanReversionStrategy(window=5, std_multiplier=1.0))

    # Giả lập một BacktestResult bị lỗi Engine: Fill xảy ra ngay tại bar_index == created_bar_index
    fake_orders = pd.DataFrame([{"order_id": "ORD-FAKE", "origin": OrderOrigin.STRATEGY_ENTRY.value}])
    fake_events = pd.DataFrame(
        [
            {"order_id": "ORD-FAKE", "event_type": "CREATED", "bar_index": 6, "timestamp": df["time"].iloc[6]},
            {"order_id": "ORD-FAKE", "event_type": "FILLED", "bar_index": 6, "timestamp": df["time"].iloc[6]},
        ]
    )
    res.orders = fake_orders
    res.order_events = fake_events

    violations = audit_order_eligibility(res)
    assert len(violations) == 1
    assert violations[0].check_type == "ORDER_ELIGIBILITY"
    assert violations[0].source == "ENGINE"
    assert violations[0].bar_index == 6


def test_sampling_is_deterministic_with_seed():
    df = _make_market(250)
    vcfg = ValidationConfig(random_seed=99, sample_step=3, max_lookahead_samples=12, check_backtest_state=False)
    strat = MeanReversionStrategy(window=10, std_multiplier=1.5)

    r1 = verify_no_lookahead(df, strat, validation_config=vcfg)
    r2 = verify_no_lookahead(df, strat, validation_config=vcfg)
    assert r1.checked_bars == r2.checked_bars
    assert len(r1.checked_bars) == 12
