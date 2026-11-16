"""Unit tests cho H5 – Determinism, Seed Management & Reproducibility (Giai đoạn 3)."""
from __future__ import annotations

import copy
import random
from typing import ClassVar
import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.validation.config import ValidationConfig
from quant_backtester.validation.models import ValidationStatus
from quant_backtester.validation.reproducibility import (
    classify_fingerprint_pair,
    compute_input_fingerprint,
    compute_result_fingerprint,
    deterministic_seed_context,
    verify_reproducibility,
)


def _make_data(n: int = 60, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 2000.0 + np.cumsum(rng.normal(0.0, 2.5, size=n))
    o = np.r_[2000.0, c[:-1]]
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-02-01", periods=n, freq="15min", tz="UTC"),
            "open": o,
            "high": np.maximum(o, c) + 1.0,
            "low": np.minimum(o, c) - 1.0,
            "close": c,
            "volume": [100.0] * n,
        }
    )


class SeededRandomStrategy(Strategy):
    """Strategy dùng RNG nội bộ; nếu nhận set_rng(rng) thì sẽ tái lập theo seed."""

    name = "seeded_random"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    def __init__(self, p_trade: float = 0.25):
        super().__init__(p_trade=p_trade)
        self.p_trade = p_trade
        self.rng = None

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def on_bar(self, bar: pd.Series, context) -> Signal | None:
        val = self.rng.random() if self.rng is not None else np.random.random()
        if context.position.is_flat and val < self.p_trade:
            return Signal(time=pd.Timestamp(bar["time"]), symbol=context.symbol, signal_type=SignalType.LONG)
        if not context.position.is_flat and val > 1.0 - self.p_trade:
            return Signal(time=pd.Timestamp(bar["time"]), symbol=context.symbol, signal_type=SignalType.EXIT_LONG)
        return None


class UncontrolledNonDetStrategy(Strategy):
    """Strategy cố tình không dùng seed và tự thay đổi theo counter toàn cục."""

    name = "uncontrolled_nondet"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN
    _GLOBAL_CALLS = 0

    def on_bar(self, bar: pd.Series, context) -> Signal | None:
        UncontrolledNonDetStrategy._GLOBAL_CALLS += 1
        if (UncontrolledNonDetStrategy._GLOBAL_CALLS + context.bar_index) % 7 == 0 and context.position.is_flat:
            return Signal(time=pd.Timestamp(bar["time"]), symbol=context.symbol, signal_type=SignalType.LONG)
        if not context.position.is_flat and context.bar_index % 4 == 0:
            return Signal(time=pd.Timestamp(bar["time"]), symbol=context.symbol, signal_type=SignalType.EXIT_LONG)
        return None


def test_input_fingerprint_identical_for_same_inputs_and_changes_on_data_or_fee():
    df = _make_data(40)
    cfg = BacktestConfig(commission_per_lot=5.0)
    strat = MeanReversionStrategy(20, 2.0)

    fp1 = compute_input_fingerprint(df, strat, cfg, seed=42)
    fp2 = compute_input_fingerprint(df.copy(), MeanReversionStrategy(20, 2.0), copy.deepcopy(cfg), seed=42)
    assert fp1 == fp2

    df_mod = df.copy()
    df_mod.loc[5, "close"] += 0.01
    assert compute_input_fingerprint(df_mod, strat, cfg, seed=42) != fp1

    cfg_mod = BacktestConfig(commission_per_lot=7.0)
    assert compute_input_fingerprint(df, strat, cfg_mod, seed=42) != fp1


def test_result_fingerprint_detects_mutations_in_equity_trades_and_ledger():
    df = _make_data(60)
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0)
    rep = verify_reproducibility(df, MeanReversionStrategy(10, 1.5), cfg, seed=42, n_runs=3)
    assert rep.is_reproducible
    assert rep.status == ValidationStatus.PASS
    assert rep.classification == "SAME_INPUT_SAME_RESULT"

    # Chạy 1 lần lấy result để kiểm tra nhạy cảm từng thành phần của Result Fingerprint
    from quant_backtester.validation.grid_runner import run_backtest

    base = run_backtest(df, MeanReversionStrategy(10, 1.5), cfg)
    base_fp = compute_result_fingerprint(base)

    mod_eq = copy.deepcopy(base)
    mod_eq.equity.loc[mod_eq.equity.index[-1], "equity"] += 0.5
    assert compute_result_fingerprint(mod_eq) != base_fp

    mod_tr = copy.deepcopy(base)
    if not mod_tr.trades.empty:
        mod_tr.trades.loc[0, "net_pnl"] += 1.0
        assert compute_result_fingerprint(mod_tr) != base_fp

    mod_led = copy.deepcopy(base)
    if not mod_led.ledger.empty:
        mod_led.ledger.loc[0, "amount"] += 0.25
        assert compute_result_fingerprint(mod_led) != base_fp


def test_result_fingerprint_handles_column_order_timezone_and_nan_inf():
    df = _make_data(30)
    from quant_backtester.validation.grid_runner import run_backtest

    res_a = run_backtest(df, MeanReversionStrategy(10, 1.5), BacktestConfig())
    res_b = copy.deepcopy(res_a)
    # Đảo thứ tự cột trong equity & đổi timezone UTC -> Asia/Ho_Chi_Minh (cùng thời điểm tuyệt đối)
    res_b.equity = res_b.equity[list(reversed(res_b.equity.columns))].copy()
    res_b.equity["time"] = pd.to_datetime(res_b.equity["time"]).dt.tz_convert("Asia/Ho_Chi_Minh")
    res_a.metrics["nan_metric"] = float("nan")
    res_b.metrics["nan_metric"] = float("nan")
    res_a.metrics["inf_metric"] = float("inf")
    res_b.metrics["inf_metric"] = float("inf")

    assert compute_result_fingerprint(res_a) == compute_result_fingerprint(res_b)


def test_seeded_random_strategy_reproduces_and_restores_global_rng_state():
    random.seed(12345)
    np.random.seed(12345)
    before_py = random.random()
    before_np = float(np.random.random())

    # Đặt lại về cùng điểm xuất phát để đối chiếu sau khi chạy deterministic_seed_context
    random.seed(12345)
    np.random.seed(12345)
    _ = random.random()
    _ = float(np.random.random())
    expected_next_py = random.random()
    expected_next_np = float(np.random.random())

    random.seed(12345)
    np.random.seed(12345)
    assert random.random() == before_py
    assert float(np.random.random()) == before_np

    with deterministic_seed_context(999) as gen:
        assert abs(gen.random() - np.random.default_rng(999).random()) < 1e-12
        _ = random.random()
        _ = np.random.random()

    # Global RNG state bên ngoài phải được khôi phục nguyên vẹn
    assert random.random() == expected_next_py
    assert float(np.random.random()) == expected_next_np

    rep = verify_reproducibility(_make_data(50), SeededRandomStrategy(0.3), BacktestConfig(), seed=77)
    assert rep.is_reproducible


def test_detects_non_deterministic_strategy():
    UncontrolledNonDetStrategy._GLOBAL_CALLS = 0
    rep = verify_reproducibility(_make_data(40), UncontrolledNonDetStrategy(), BacktestConfig(), n_runs=2)
    assert not rep.is_reproducible
    assert rep.status == ValidationStatus.FAIL
    assert rep.classification == "SAME_INPUT_DIFFERENT_RESULT"
    assert len(rep.differences) > 0
    assert classify_fingerprint_pair("a", "b", "x", "x") == "DIFFERENT_INPUT_SAME_RESULT"
