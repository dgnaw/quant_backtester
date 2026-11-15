"""
Unit tests for Tasks G4 & G5: Annualization Factor, CAGR, Volatility, Sharpe,
Sortino, Calmar & Return Distribution Metrics.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from quant_backtester.analytics.annualization import (
    bars_per_year_from_timeframe,
    infer_bars_per_year_from_timestamps,
    resolve_bars_per_year,
)
from quant_backtester.analytics.performance import (
    annual_to_bar_rate,
    calculate_annualized_volatility,
    calculate_cagr_pct,
    calculate_calmar_ratio,
    calculate_downside_deviation,
    calculate_return_distribution_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from quant_backtester.config import EngineConfig
from quant_backtester.data.trading_calendar import TradingCalendar


def test_1_timeframe_table_m1_to_w1():
    assert bars_per_year_from_timeframe("M1") == 362_880
    assert bars_per_year_from_timeframe("M5") == 72_576
    assert bars_per_year_from_timeframe("M15") == 24_192
    assert bars_per_year_from_timeframe("M30") == 12_096
    assert bars_per_year_from_timeframe("H1") == 6_048
    assert bars_per_year_from_timeframe("H4") == 1_512
    assert bars_per_year_from_timeframe("D1") == 252
    assert bars_per_year_from_timeframe("W1") == 52


def test_2_config_override_and_timeframe_resolution():
    cfg = EngineConfig(bars_per_year=252)
    assert resolve_bars_per_year(config=cfg) == 252
    assert resolve_bars_per_year(config=cfg, timeframe="H1") == 6_048

    cfg_tf = EngineConfig(timeframe="H4")
    assert resolve_bars_per_year(config=cfg_tf) == 1_512


def test_3_trading_calendar_daily_break_adjustment():
    # 1 hour daily break (23 active hours/day) on H1 -> 23 * 252 = 5,796 bars/year
    cal = TradingCalendar(daily_break_start_hour=22, daily_break_end_hour=23)
    assert bars_per_year_from_timeframe("H1", trading_calendar=cal) == 23 * 252


def test_4_infer_bars_per_year_from_timestamps_with_weekend_gaps():
    # Create 15-minute bars across a weekend gap; median delta is still 15 minutes -> 24,192
    part1 = pd.date_range("2025-01-03 20:00:00", periods=8, freq="15min")
    part2 = pd.date_range("2025-01-06 00:00:00", periods=8, freq="15min")
    ts = pd.Series(list(part1) + list(part2))
    assert infer_bars_per_year_from_timestamps(ts) == 24_192

    cfg_auto = EngineConfig(auto_annualize_from_data=True)
    assert resolve_bars_per_year(config=cfg_auto, time_series=ts) == 24_192


def test_5_cagr_without_cash_flow():
    # Double in 2 years (504 daily bars with 252 bars/year) -> CAGR = (2.0 ** 0.5 - 1) * 100 = 41.421356%
    cagr = calculate_cagr_pct(2.0, n_return_bars=504, bars_per_year=252)
    assert cagr is not None
    assert math.isclose(cagr, (math.sqrt(2.0) - 1.0) * 100.0, rel_tol=1e-9)


def test_6_cagr_with_wealth_index_series_and_invalid_growth():
    wealth = pd.Series([1.0, 1.10, 1.21])  # 2 years at 1 bar/year -> 10% CAGR
    cagr = calculate_cagr_pct(wealth, bars_per_year=1)
    assert cagr is not None
    assert math.isclose(cagr, 10.0, rel_tol=1e-9)

    # Zero or negative growth factor -> undefined power law -> returns None
    assert calculate_cagr_pct(0.0, n_return_bars=252, bars_per_year=252) is None
    assert calculate_cagr_pct(-0.5, n_return_bars=252, bars_per_year=252) is None
    assert calculate_cagr_pct(1.2, n_return_bars=0, bars_per_year=252) is None


def test_7_sharpe_ratio_with_zero_risk_free_rate():
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    expected = (mean_r / std_r) * math.sqrt(252.0)
    actual = calculate_sharpe_ratio(returns, bars_per_year=252, risk_free_rate_annual=0.0)
    assert actual is not None
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_8_sharpe_ratio_with_positive_risk_free_rate():
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    rf_annual = 0.05
    rf_bar = annual_to_bar_rate(rf_annual, 252)
    excess = np.array(returns) - rf_bar
    expected = (float(np.mean(excess)) / float(np.std(excess, ddof=1))) * math.sqrt(252.0)
    actual = calculate_sharpe_ratio(returns, bars_per_year=252, risk_free_rate_annual=rf_annual)
    assert actual is not None
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_9_annualized_volatility():
    returns = [0.01, -0.02, 0.015, -0.005, 0.02]
    expected = float(np.std(returns, ddof=1)) * math.sqrt(252.0)
    actual = calculate_annualized_volatility(returns, bars_per_year=252)
    assert actual is not None
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_10_constant_return_series_sharpe_policy():
    # Zero excess return & zero std -> 0.0
    assert calculate_sharpe_ratio([0.0, 0.0, 0.0], bars_per_year=252) == 0.0
    # Positive constant excess return & zero std -> +inf
    s_pos = calculate_sharpe_ratio([0.01, 0.01, 0.01], bars_per_year=252)
    assert s_pos is not None and math.isinf(s_pos) and s_pos > 0
    # Negative constant excess return & zero std -> -inf
    s_neg = calculate_sharpe_ratio([-0.01, -0.01, -0.01], bars_per_year=252)
    assert s_neg is not None and math.isinf(s_neg) and s_neg < 0


def test_11_hand_calculated_downside_deviation_lpm2():
    # Returns: [+0.02, -0.01, +0.01, -0.03] -> downside diffs: [0, -0.01, 0, -0.03]
    # Squared: [0, 0.0001, 0, 0.0009] -> mean over N=4 is 0.00025 -> sqrt = 0.0158113883
    returns = [0.02, -0.01, 0.01, -0.03]
    dd_bar = calculate_downside_deviation(returns, bars_per_year=252, annualize=False)
    dd_ann = calculate_downside_deviation(returns, bars_per_year=252, annualize=True)
    assert dd_bar is not None and dd_ann is not None
    assert math.isclose(dd_bar, math.sqrt(0.00025), rel_tol=1e-9)
    assert math.isclose(dd_ann, math.sqrt(0.00025) * math.sqrt(252.0), rel_tol=1e-9)


def test_12_sortino_ratio_with_wins_and_losses():
    returns = [0.02, -0.01, 0.01, -0.03]
    expected_dd_bar = math.sqrt(0.00025)
    expected_sortino = (float(np.mean(returns)) / expected_dd_bar) * math.sqrt(252.0)
    actual = calculate_sortino_ratio(returns, bars_per_year=252, risk_free_rate_annual=0.0)
    assert actual is not None
    assert math.isclose(actual, expected_sortino, rel_tol=1e-9)


def test_13_sortino_ratio_without_downside():
    s_inf = calculate_sortino_ratio([0.01, 0.02, 0.015], bars_per_year=252)
    assert s_inf is not None and math.isinf(s_inf) and s_inf > 0
    assert calculate_sortino_ratio([0.0, 0.0, 0.0], bars_per_year=252) == 0.0


def test_14_calmar_ratio_valid_and_zero_drawdown():
    assert math.isclose(calculate_calmar_ratio(20.0, -10.0) or 0.0, 2.0, rel_tol=1e-9)
    # Zero max drawdown -> Calmar is None per DG5.E ("Chỉ tính khi Max Drawdown khác 0 và CAGR hợp lệ")
    assert calculate_calmar_ratio(20.0, 0.0) is None
    assert calculate_calmar_ratio(None, -10.0) is None


def test_15_skewness_excess_kurtosis_and_tail_ratio():
    rng = [0.05, -0.01, 0.02, -0.02, 0.01, 0.03, -0.015, 0.025, 0.005, -0.005]
    dist = calculate_return_distribution_metrics(rng)
    s = pd.Series(rng, dtype=float)
    assert dist["return_skewness"] is not None
    assert math.isclose(dist["return_skewness"], float(s.skew()), rel_tol=1e-9)
    # Must be Fisher's Excess Kurtosis (normal = 0.0), matching pandas Series.kurt()
    assert dist["return_kurtosis"] is not None
    assert math.isclose(dist["return_kurtosis"], float(s.kurt()), rel_tol=1e-9)
    assert dist["return_kurtosis"] == dist["return_excess_kurtosis"]
    assert dist["tail_ratio"] is not None and dist["tail_ratio"] > 0.0


def test_16_insufficient_observations_and_invalid_inputs():
    assert calculate_sharpe_ratio([0.01], bars_per_year=252) is None
    assert calculate_sortino_ratio([0.01], bars_per_year=252) is None
    assert calculate_annualized_volatility([0.01], bars_per_year=252) is None
    dist_short = calculate_return_distribution_metrics([0.01, -0.01])
    assert dist_short["return_skewness"] is None
    assert dist_short["return_kurtosis"] is None
    assert dist_short["tail_ratio"] is None

    with pytest.raises(ValueError):
        annual_to_bar_rate(-1.0, 252)
    with pytest.raises(ValueError):
        bars_per_year_from_timeframe("UNKNOWN_TF")
