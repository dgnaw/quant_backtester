from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_backtester.analytics.benchmark import (
    BENCHMARK_SERIES_COLUMNS,
    build_buy_and_hold_benchmark,
    calculate_benchmark_metrics,
    prepare_external_benchmark,
)
from quant_backtester.analytics.period_analysis import (
    MONTHLY_MATRIX_COLUMNS,
    PERIOD_STATS_COLUMNS,
    REGIME_STATS_COLUMNS,
    analyze_by_period,
    analyze_custom_regimes,
    build_monthly_return_matrix,
)
from quant_backtester.analytics.report import (
    build_analytics_result,
    calculate_metrics,
    print_report,
)
from quant_backtester.config import AccountConfig, BacktestConfig, EngineConfig


# ==============================================================================
# 1. BUY & HOLD BENCHMARK & EXTERNAL BENCHMARK ALIGNMENT
# ==============================================================================


def test_build_buy_and_hold_benchmark_respects_effective_trade_start_warmup():
    times = pd.date_range("2025-01-01 00:00:00", periods=5, freq="1h", tz="UTC")
    market_df = pd.DataFrame(
        {
            "time": times,
            "close": [100.0, 120.0, 200.0, 220.0, 240.0],
        }
    )
    # Warm-up 2 bars => effective_trade_start = 2 (tại close = 200.0)
    bench = build_buy_and_hold_benchmark(
        market_df, initial_balance=1_000.0, effective_trade_start=2
    )
    assert list(bench.columns) == BENCHMARK_SERIES_COLUMNS
    # Bar 0, 1, 2 phải giữ nguyên initial_balance = 1000.0
    assert bench["equity"].iloc[0] == pytest.approx(1_000.0)
    assert bench["equity"].iloc[1] == pytest.approx(1_000.0)
    assert bench["equity"].iloc[2] == pytest.approx(1_000.0)
    # Bar 3: 1000 * (220 / 200) = 1100
    assert bench["equity"].iloc[3] == pytest.approx(1_100.0)
    # Bar 4: 1000 * (240 / 200) = 1200
    assert bench["equity"].iloc[4] == pytest.approx(1_200.0)
    assert bench["cumulative_return"].iloc[4] == pytest.approx(0.20)


def test_prepare_external_benchmark_aligns_subset_timestamps_and_scales_equity():
    strat_times = pd.date_range("2025-01-01 00:00:00", periods=4, freq="1D", tz="UTC")
    # External benchmark có thêm ngày trước và sau
    ext_times = pd.date_range("2024-12-31 00:00:00", periods=6, freq="1D", tz="UTC")
    ext_df = pd.DataFrame(
        {
            "time": ext_times,
            "close": [90.0, 100.0, 105.0, 110.0, 120.0, 130.0],
        }
    )
    aligned = prepare_external_benchmark(
        ext_df, target_times=strat_times, initial_balance=1_000.0
    )
    assert len(aligned) == 4
    assert list(aligned["time"]) == list(strat_times)
    # Quy chuẩn về 1000 tại mốc đầu tiên khớp (close = 100)
    assert aligned["equity"].iloc[0] == pytest.approx(1_000.0)
    assert aligned["equity"].iloc[-1] == pytest.approx(1_200.0)


def test_prepare_external_benchmark_supports_series_and_return_column():
    times = pd.date_range("2025-01-01", periods=3, freq="1D", tz="UTC")
    series_bench = pd.Series([50.0, 55.0, 60.5], index=times)
    out_s = prepare_external_benchmark(series_bench, initial_balance=1_000.0)
    assert out_s["equity"].tolist() == pytest.approx([1_000.0, 1_100.0, 1_210.0])

    ret_bench = pd.DataFrame({"time": times, "return": [0.0, 0.10, -0.05]})
    out_r = prepare_external_benchmark(ret_bench, initial_balance=1_000.0)
    assert out_r["equity"].tolist() == pytest.approx([1_000.0, 1_100.0, 1_045.0])


# ==============================================================================
# 2. BENCHMARK METRICS: BETA, ALPHA, ACTIVE RETURN, TRACKING ERROR, IR, CAPTURE
# ==============================================================================


def test_calculate_benchmark_metrics_identical_series_gives_beta_one_alpha_zero():
    times = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
    eq = [1_000.0, 1_020.0, 1_010.0, 1_040.0, 1_030.0, 1_060.0]
    strat_df = pd.DataFrame({"time": times, "equity": eq})
    bench_df = pd.DataFrame({"time": times, "equity": eq})

    m = calculate_benchmark_metrics(
        strat_df, bench_df, bars_per_year=252, risk_free_rate_annual=0.0
    )
    assert m["beta"] == pytest.approx(1.0, rel=1e-9)
    assert m["alpha_annual_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m["active_return_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m["tracking_error_annual"] == pytest.approx(0.0, abs=1e-9)
    assert m["information_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert m["up_capture_pct"] == pytest.approx(100.0, rel=1e-9)
    assert m["down_capture_pct"] == pytest.approx(100.0, rel=1e-9)


def test_calculate_benchmark_metrics_2x_leveraged_plus_constant_alpha():
    times = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
    r_b = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    alpha_bar = 0.001  # +0.1% mỗi bar
    r_s = 2.0 * r_b + alpha_bar

    eq_b = [1_000.0]
    eq_s = [1_000.0]
    for rb, rs in zip(r_b, r_s):
        eq_b.append(eq_b[-1] * (1.0 + rb))
        eq_s.append(eq_s[-1] * (1.0 + rs))

    strat_df = pd.DataFrame({"time": times, "equity": eq_s})
    bench_df = pd.DataFrame({"time": times, "equity": eq_b})

    m = calculate_benchmark_metrics(
        strat_df, bench_df, bars_per_year=252, risk_free_rate_annual=0.0
    )
    assert m["beta"] == pytest.approx(2.0, rel=1e-9)
    assert m["alpha_annual_pct"] == pytest.approx(alpha_bar * 252 * 100.0, rel=1e-9)
    assert m["active_return_pct"] > 0.0
    assert m["tracking_error_annual"] > 0.0
    assert m["information_ratio"] > 0.0


def test_calculate_benchmark_metrics_up_and_down_capture_geometric():
    times = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    # Benchmark: +10%, -10%, +10%, -10%
    r_b = [0.10, -0.10, 0.10, -0.10]
    # Strategy: +15% khi benchmark tăng, -5% khi benchmark giảm
    r_s = [0.15, -0.05, 0.15, -0.05]

    eq_b = [1_000.0]
    eq_s = [1_000.0]
    for rb, rs in zip(r_b, r_s):
        eq_b.append(eq_b[-1] * (1.0 + rb))
        eq_s.append(eq_s[-1] * (1.0 + rs))

    strat_df = pd.DataFrame({"time": times, "equity": eq_s})
    bench_df = pd.DataFrame({"time": times, "equity": eq_b})

    m = calculate_benchmark_metrics(strat_df, bench_df, bars_per_year=252)
    assert m["up_capture_pct"] == pytest.approx(150.0, rel=1e-9)
    assert m["down_capture_pct"] == pytest.approx(50.0, rel=1e-9)


def test_calculate_benchmark_metrics_empty_or_none_returns_none_not_fake_zeros():
    times = pd.date_range("2025-01-01", periods=4, freq="1D", tz="UTC")
    strat_df = pd.DataFrame({"time": times, "equity": [1000.0, 1010.0, 1020.0, 1030.0]})

    m_none = calculate_benchmark_metrics(strat_df, None)
    assert all(v is None for v in m_none.values())

    m_empty = calculate_benchmark_metrics(strat_df, pd.DataFrame())
    assert all(v is None for v in m_empty.values())


# ==============================================================================
# 3. PERIOD PERFORMANCE ANALYSIS (MONTHLY, QUARTERLY, YEARLY, MATRIX, REGIMES)
# ==============================================================================


def test_analyze_by_period_monthly_quarterly_yearly_and_trade_bucketing():
    times = pd.to_datetime(
        [
            "2025-01-01 00:00:00",
            "2025-01-15 00:00:00",
            "2025-01-31 00:00:00",
            "2025-02-15 00:00:00",
            "2025-02-28 00:00:00",
            "2025-04-15 00:00:00",
        ],
        utc=True,
    )
    # Jan: 1000 -> 1050 -> 1100 (+10%)
    # Feb: 1100 -> 1045 -> 1210 (+10%)
    # Apr (Q2): 1210 -> 1331 (+10%)
    equity_df = pd.DataFrame(
        {
            "time": times,
            "equity": [1_000.0, 1_050.0, 1_100.0, 1_045.0, 1_210.0, 1_331.0],
        }
    )
    trades_df = pd.DataFrame(
        [
            {
                "trade_id": "TRD-1",
                "exit_time": pd.Timestamp("2025-01-20", tz="UTC"),
                "net_pnl": 100.0,
                "gross_pnl": 102.0,
            },
            {
                "trade_id": "TRD-2",
                "exit_time": pd.Timestamp("2025-02-10", tz="UTC"),
                "net_pnl": -55.0,
                "gross_pnl": -50.0,
            },
            {
                "trade_id": "TRD-3",
                "exit_time": pd.Timestamp("2025-02-25", tz="UTC"),
                "net_pnl": 165.0,
                "gross_pnl": 170.0,
            },
        ]
    )

    m_stats = analyze_by_period(equity_df, trades_df, freq="ME", initial_balance=1_000.0)
    assert list(m_stats.columns) == PERIOD_STATS_COLUMNS
    assert m_stats["period"].tolist() == ["2025-01", "2025-02", "2025-04"]
    assert m_stats["return_pct"].tolist() == pytest.approx([10.0, 10.0, 10.0], rel=1e-9)
    assert m_stats["trades"].tolist() == [1, 2, 0]
    assert m_stats["net_pnl"].tolist() == pytest.approx([100.0, 110.0, 0.0])
    # Tháng 2 giảm từ 1100 xuống 1045 (-5%) trước khi lên 1210
    assert m_stats.loc[1, "max_drawdown_pct"] == pytest.approx(-5.0, rel=1e-9)

    q_stats = analyze_by_period(equity_df, trades_df, freq="QE", initial_balance=1_000.0)
    assert q_stats["period"].tolist() == ["2025-Q1", "2025-Q2"]
    # Q1 ghép lãi Jan (+10%) và Feb (+10%) => +21%
    assert q_stats.loc[0, "return_pct"] == pytest.approx(21.0, rel=1e-9)
    assert q_stats.loc[1, "return_pct"] == pytest.approx(10.0, rel=1e-9)

    y_stats = analyze_by_period(equity_df, trades_df, freq="YE", initial_balance=1_000.0)
    assert y_stats["period"].tolist() == ["2025"]
    assert y_stats.loc[0, "return_pct"] == pytest.approx(33.1, rel=1e-9)
    assert y_stats.loc[0, "trades"] == 3


def test_analyze_by_period_eliminates_external_deposit_via_twr():
    times = pd.to_datetime(
        ["2025-01-01", "2025-01-31", "2025-02-28"], utc=True
    )
    # Jan: 1000 -> 1100 (+10%)
    # Feb: nạp thêm 500, nhưng không có lãi giao dịch => equity = 1600 (TWR = 0%)
    equity_df = pd.DataFrame(
        {"time": times, "equity": [1_000.0, 1_100.0, 1_600.0]}
    )
    ledger_df = pd.DataFrame(
        [
            {
                "entry_id": "LED-00001",
                "time": times[0],
                "transaction_type": "DEPOSIT",
                "amount": 1_000.0,
                "reason": "INITIAL_BALANCE",
            },
            {
                "entry_id": "LED-00002",
                "time": times[2],
                "transaction_type": "DEPOSIT",
                "amount": 500.0,
                "reason": "TOP_UP",
            },
        ]
    )
    m_stats = analyze_by_period(
        equity_df, ledger_df=ledger_df, freq="ME", initial_balance=1_000.0
    )
    assert m_stats["return_pct"].tolist() == pytest.approx([10.0, 0.0], abs=1e-9)


def test_build_monthly_return_matrix_compounded_ytd():
    times = pd.to_datetime(
        ["2025-01-01", "2025-01-31", "2025-02-28", "2025-03-31"], utc=True
    )
    # Jan: +10% (1000 -> 1100), Feb: -10% (1100 -> 990), Mar: +20% (990 -> 1188)
    # YTD = (1.1 * 0.9 * 1.2 - 1) * 100 = +18.8% (không phải 10 - 10 + 20 = 20%)
    equity_df = pd.DataFrame(
        {"time": times, "equity": [1_000.0, 1_100.0, 990.0, 1_188.0]}
    )
    mat = build_monthly_return_matrix(equity_df, initial_balance=1_000.0)
    assert list(mat.columns) == MONTHLY_MATRIX_COLUMNS
    assert len(mat) == 1
    row = mat.iloc[0]
    assert row["year"] == 2025
    assert row["Jan"] == pytest.approx(10.0)
    assert row["Feb"] == pytest.approx(-10.0)
    assert row["Mar"] == pytest.approx(20.0)
    assert pd.isna(row["Apr"])
    assert row["YTD"] == pytest.approx(18.8, rel=1e-9)


def test_analyze_custom_regimes_bull_and_bear_windows():
    times = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
    # Bar 0..2 (Bull): 1000 -> 1100 -> 1210 (+21%)
    # Bar 3..5 (Bear): 1210 -> 1089 (-10%) -> 980.1 (-19% so với đỉnh 1210)
    equity_df = pd.DataFrame(
        {
            "time": times,
            "equity": [1_000.0, 1_100.0, 1_210.0, 1_089.0, 1_089.0, 980.1],
        }
    )
    trades_df = pd.DataFrame(
        [
            {"exit_time": times[2], "net_pnl": 210.0, "gross_pnl": 215.0},
            {"exit_time": times[5], "net_pnl": -229.9, "gross_pnl": -225.0},
        ]
    )
    regimes = [
        ("Bull_Phase", times[0], times[2]),
        ("Bear_Phase", times[3], times[5]),
    ]
    reg_df = analyze_custom_regimes(
        equity_df, trades_df, regimes=regimes, initial_balance=1_000.0
    )
    assert list(reg_df.columns) == REGIME_STATS_COLUMNS
    assert reg_df["regime"].tolist() == ["Bull_Phase", "Bear_Phase"]
    assert reg_df.loc[0, "return_pct"] == pytest.approx(21.0, rel=1e-9)
    assert reg_df.loc[0, "trades"] == 1
    assert reg_df.loc[1, "return_pct"] == pytest.approx(-19.0, rel=1e-9)
    assert reg_df.loc[1, "max_drawdown_pct"] == pytest.approx(-19.0, rel=1e-9)
    assert reg_df.loc[1, "trades"] == 1


def test_analyze_custom_regimes_rejects_inverted_window():
    times = pd.date_range("2025-01-01", periods=3, freq="1D", tz="UTC")
    equity_df = pd.DataFrame({"time": times, "equity": [1000.0, 1010.0, 1020.0]})
    with pytest.raises(ValueError, match="start_time"):
        analyze_custom_regimes(
            equity_df, None, regimes=[("Invalid", times[2], times[0])]
        )


# ==============================================================================
# 4. BACKWARD COMPATIBILITY, PRINT REPORT & DETERMINISM
# ==============================================================================


def test_calculate_metrics_preserves_all_11_legacy_keys_and_types():
    times = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    result_dict = {
        "equity": pd.DataFrame(
            {"time": times, "equity": [1_000.0, 1_000.0, 1_050.0, 1_020.0, 1_080.0]}
        ),
        "trades": pd.DataFrame(
            [
                {"net_pnl": 50.0, "gross_pnl": 52.0, "exit_time": times[2]},
                {"net_pnl": -30.0, "gross_pnl": -28.0, "exit_time": times[3]},
                {"net_pnl": 60.0, "gross_pnl": 62.0, "exit_time": times[4]},
            ]
        ),
    }
    cfg = BacktestConfig(initial_balance=1_000.0)
    m = calculate_metrics(result_dict, cfg)

    legacy_keys = [
        "initial_balance",
        "final_equity",
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe_annual",
        "total_trades",
        "wins",
        "losses",
        "win_rate_pct",
        "profit_factor",
        "expectancy_cash_per_trade",
    ]
    for k in legacy_keys:
        assert k in m

    assert m["initial_balance"] == pytest.approx(1_000.0)
    assert m["final_equity"] == pytest.approx(1_080.0)
    assert m["total_return_pct"] == pytest.approx(8.0)
    assert m["total_trades"] == 3
    assert m["wins"] == 2
    assert m["losses"] == 1


def test_zero_trades_and_single_bar_equity_do_not_crash():
    times = pd.date_range("2025-01-01", periods=1, freq="15min", tz="UTC")
    res = {
        "equity": pd.DataFrame({"time": times, "equity": [1_000.0]}),
        "trades": pd.DataFrame(),
    }
    m = calculate_metrics(res, BacktestConfig(initial_balance=1_000.0))
    assert m["total_trades"] == 0
    assert m["sharpe_annual"] == 0.0
    assert m["sharpe_is_valid"] is False
    assert m["cagr_pct"] is None


def test_print_report_handles_none_and_inf_gracefully(capsys):
    times = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    res = {
        "equity": pd.DataFrame(
            {"time": times, "equity": [1_000.0, 1_010.0, 1_020.0, 1_030.0]}
        ),
        "trades": pd.DataFrame([{"net_pnl": 30.0, "gross_pnl": 30.0}]),
    }
    m = calculate_metrics(res, BacktestConfig(initial_balance=1_000.0))
    text = print_report(m, title="UNIT TEST REPORT")
    captured = capsys.readouterr().out
    assert "UNIT TEST REPORT" in text
    assert "UNIT TEST REPORT" in captured
    assert "+inf" in text  # Profit factor khi chỉ có lệnh thắng


def test_build_analytics_result_does_not_mutate_inputs_and_is_deterministic():
    times = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
    eq_orig = pd.DataFrame(
        {"time": times, "equity": [1_000.0, 1_020.0, 1_010.0, 1_050.0, 1_040.0, 1_080.0]}
    )
    tr_orig = pd.DataFrame(
        [
            {"exit_time": times[1], "net_pnl": 20.0, "gross_pnl": 22.0},
            {"exit_time": times[2], "net_pnl": -10.0, "gross_pnl": -8.0},
            {"exit_time": times[5], "net_pnl": 70.0, "gross_pnl": 72.0},
        ]
    )
    eq_copy = eq_orig.copy(deep=True)
    tr_copy = tr_orig.copy(deep=True)

    cfg = EngineConfig(account=AccountConfig(initial_balance=1_000.0))
    res1 = build_analytics_result({"equity": eq_orig, "trades": tr_orig}, cfg)
    res2 = build_analytics_result({"equity": eq_orig, "trades": tr_orig}, cfg)

    pd.testing.assert_frame_equal(eq_orig, eq_copy)
    pd.testing.assert_frame_equal(tr_orig, tr_copy)
    assert res1.metrics == res2.metrics
    pd.testing.assert_frame_equal(res1.returns_series, res2.returns_series)
    pd.testing.assert_frame_equal(res1.drawdown_series, res2.drawdown_series)
