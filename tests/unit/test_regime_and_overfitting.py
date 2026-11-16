"""Unit tests cho H8 – Market Regime Analysis & H9 – Overfitting / Multiple Testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.validation.config import MarketRegimeConfig, OverfittingThresholds
from quant_backtester.validation.models import ValidationStatus
from quant_backtester.validation.overfitting import (
    adjust_multiple_testing_pvalues,
    calculate_deflated_sharpe_ratio,
    calculate_probabilistic_sharpe_ratio,
    calculate_spearman_rank_correlation,
    evaluate_overfitting,
    expected_maximum_sharpe_ratio,
)
from quant_backtester.validation.regime_analysis import (
    analyze_market_regimes,
    classify_market_regimes,
    extract_contiguous_segments,
)


def _make_multi_regime_market() -> pd.DataFrame:
    # 40 nến tăng đều (Bull), 40 nến đi ngang biên độ hẹp (Sideways Low Vol), 40 nến giảm mạnh biến động cao (Bear High Vol)
    rng = np.random.default_rng(42)
    r_bull = np.full(40, 0.008) + rng.normal(0.0, 0.001, 40)
    r_side = rng.normal(0.0, 0.0005, 40)
    r_bear = np.full(40, -0.01) + rng.normal(0.0, 0.015, 40)
    rets = np.concatenate([r_bull, r_side, r_bear])
    closes = 2000.0 * np.exp(np.cumsum(rets))
    opens = np.r_[2000.0, closes[:-1]]
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=len(closes), freq="1D", tz="UTC"),
            "open": opens,
            "high": np.maximum(opens, closes) + 5.0,
            "low": np.minimum(opens, closes) - 5.0,
            "close": closes,
            "volume": [100.0] * len(closes),
        }
    )


def test_regime_classification_trend_volatility_causality_and_segments():
    df = _make_multi_regime_market()
    rcfg = MarketRegimeConfig(trend_window=15, vol_window=15, trend_threshold=1.0, vol_threshold_method="EXPANDING_MEDIAN")
    labeled_full = classify_market_regimes(df, rcfg)

    # Đầu chuỗi chưa đủ cửa sổ -> UNKNOWN
    assert labeled_full["trend_regime"].iloc[0] == "UNKNOWN"
    assert labeled_full["combined_regime"].iloc[10] == "UNKNOWN"

    # Phân loại Bull, Sideways, Bear chính xác ở các vùng tương ứng
    assert labeled_full["trend_regime"].iloc[35] == "BULL_TREND"
    assert labeled_full["trend_regime"].iloc[75] == "SIDEWAYS_RANGE"
    assert labeled_full["vol_regime"].iloc[75] == "LOW_VOL"
    assert labeled_full["vol_regime"].iloc[115] == "HIGH_VOL"

    # Tính nhân quả tuyệt đối: cắt ngắn tại bar k không làm đổi nhãn từ 0..k
    k = 80
    labeled_prefix = classify_market_regimes(df.iloc[: k + 1].copy(), rcfg)
    assert labeled_full["trend_regime"].iloc[: k + 1].tolist() == labeled_prefix["trend_regime"].tolist()
    assert labeled_full["vol_regime"].iloc[: k + 1].tolist() == labeled_prefix["vol_regime"].tolist()
    assert labeled_full["combined_regime"].iloc[: k + 1].tolist() == labeled_prefix["combined_regime"].tolist()

    segs = extract_contiguous_segments(labeled_full)
    assert int(segs["bars_count"].sum()) == len(df)


def test_analyze_market_regimes_reports_return_and_exit_time_trades():
    df = _make_multi_regime_market()
    cfg = BacktestConfig(initial_balance=10_000.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    res = BacktestEngine(cfg).run(df, MeanReversionStrategy(10, 1.5))

    rep = analyze_market_regimes(df, res, MarketRegimeConfig(trend_window=15, vol_window=15))
    assert rep.status == ValidationStatus.PASS
    assert rep.trade_attribution_method == "EXIT_TIME"
    assert int(rep.trend_regime_stats["bars_count"].sum()) == len(df)
    assert int(rep.trend_regime_stats["trades"].sum()) == len(res.trades)


def test_probabilistic_sharpe_ratio_properties():
    # Observed == Benchmark => PSR == 0.5
    assert calculate_probabilistic_sharpe_ratio(0.1, benchmark_sharpe=0.1, n_observations=100) == pytest.approx(0.5)
    # Observed tăng => PSR tăng
    psr_low = calculate_probabilistic_sharpe_ratio(0.05, 0.0, n_observations=100)
    psr_high = calculate_probabilistic_sharpe_ratio(0.20, 0.0, n_observations=100)
    assert psr_low < psr_high
    # Skewness âm & fat tail (excess_kurtosis > 0) làm giảm PSR khi SR > 0
    psr_norm = calculate_probabilistic_sharpe_ratio(0.25, 0.0, n_observations=100, skewness=0.0, excess_kurtosis=0.0)
    psr_fat = calculate_probabilistic_sharpe_ratio(0.25, 0.0, n_observations=100, skewness=-1.5, excess_kurtosis=6.0)
    assert psr_fat < psr_norm
    # Thiếu quan sát -> None
    assert calculate_probabilistic_sharpe_ratio(0.2, n_observations=2) is None


def test_deflated_sharpe_ratio_and_expected_max_sharpe():
    assert expected_maximum_sharpe_ratio(1, 0.1) == pytest.approx(0.0)
    sr0_10 = expected_maximum_sharpe_ratio(10, 0.1)
    sr0_100 = expected_maximum_sharpe_ratio(100, 0.1)
    sr0_1000 = expected_maximum_sharpe_ratio(1000, 0.1)
    assert 0.0 < sr0_10 < sr0_100 < sr0_1000

    dsr_1, _ = calculate_deflated_sharpe_ratio(0.15, n_trials=1, trials_sharpe_std=0.08, n_observations=252)
    dsr_500, _ = calculate_deflated_sharpe_ratio(0.15, n_trials=500, trials_sharpe_std=0.08, n_observations=252)
    assert dsr_500 < dsr_1


def test_multiple_testing_pvalue_corrections_preserve_order():
    raw = [0.005, 0.04, 0.02, 0.30]
    bonf = adjust_multiple_testing_pvalues(raw, method="bonferroni")
    sidak = adjust_multiple_testing_pvalues(raw, method="sidak")
    holm = adjust_multiple_testing_pvalues(raw, method="holm")

    assert bonf.tolist() == pytest.approx([0.02, 0.16, 0.08, 1.0])
    assert holm.tolist() == pytest.approx([0.02, 0.08, 0.06, 0.30])
    assert (sidak >= np.array(raw) - 1e-12).all() and (sidak <= bonf + 1e-12).all()
    assert (holm <= bonf + 1e-12).all() and (holm >= np.array(raw) - 1e-12).all()


def test_spearman_rank_correlation_with_ties_and_overfitting_decision():
    assert calculate_spearman_rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert calculate_spearman_rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Có đồng hạng (ties) và toàn bộ bằng nhau -> None
    corr_tie = calculate_spearman_rank_correlation([1.0, 1.0, 2.0, 3.0], [10.0, 10.0, 20.0, 30.0])
    assert corr_tie == pytest.approx(1.0)
    assert calculate_spearman_rank_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None

    rep = evaluate_overfitting(
        observed_sharpe_annual=0.5,
        bars_per_year=252,
        n_observations=100,
        trial_sharpes_annual=[0.1, 0.2, 0.3, 0.5, 0.8],
        is_annualized_return_pct=20.0,
        oos_annualized_return_pct=2.0,
        is_sharpe_annual=2.0,
        oos_sharpe_annual=0.4,
        walk_forward_efficiency=0.1,
        is_grid_scores=[1.0, 2.0, 3.0, 4.0],
        oos_grid_scores=[4.0, 3.0, 2.0, 1.0],
        thresholds=OverfittingThresholds(min_dsr_threshold=0.95, min_wfe_threshold=0.5, min_rank_correlation=0.0, max_is_oos_degradation=0.5, fail_flag_count=2),
    )
    assert rep.status == ValidationStatus.FAIL
    assert len(rep.triggered_flags) >= 2
    assert len(rep.assumptions_and_limitations) >= 2
