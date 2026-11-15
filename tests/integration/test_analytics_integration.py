"""
End-to-End Integration tests for Group G: Accounting & Performance Analytics.
Covers R-Multiple tracking, Gate 03 Report Reconciliation, Benchmark Comparison,
Period Performance Analysis, Custom Regimes, Report Printing, and Determinism.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import pytest

from quant_backtester.analytics import (
    analyze_by_period,
    analyze_custom_regimes,
    build_analytics_result,
    calculate_metrics,
    print_report,
)
from quant_backtester.config import (
    AccountConfig,
    AnalyticsConfig,
    CommissionMode,
    EngineConfig,
    FeeConfig,
    InstrumentConfig,
    PositionMode,
    ReversePolicy,
    RiskConfig,
    SizingMode,
    SlippageConfig,
    SpreadConfig,
)
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode


def _sig(
    time: pd.Timestamp,
    symbol: str,
    signal_type: SignalType,
    quantity: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
) -> Signal:
    directives: dict[str, Any] = {}
    if quantity is not None:
        directives["quantity"] = quantity
    if stop_loss_price is not None:
        directives["stop_loss_price"] = stop_loss_price
    if take_profit_price is not None:
        directives["take_profit_price"] = take_profit_price
    meta = {"directives": directives} if directives else None
    return Signal(time=time, symbol=symbol, signal_type=signal_type, metadata=meta)


class ScriptedEventStrategy(Strategy):
    """Event-driven strategy emitting scripted signals by bar index."""

    mode = StrategyMode.EVENT_DRIVEN
    warmup_period = 0

    def __init__(self, signals_by_bar: dict[int, Signal], warmup: int = 0):
        super().__init__()
        self.signals_by_bar = signals_by_bar
        self.warmup_period = warmup

    def on_bar(self, bar: pd.Series, context) -> Signal | None:
        return self.signals_by_bar.get(context.bar_index)


def _make_ohlcv_df(
    closes: list[float],
    *,
    start: str = "2025-01-01 00:00:00",
    freq: str = "15min",
    high_offset: float = 2.0,
    low_offset: float = 2.0,
) -> pd.DataFrame:
    times = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + high_offset for o, c in zip(opens, closes)]
    lows = [min(o, c) - low_offset for o, c in zip(opens, closes)]
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


def _base_config(
    *,
    initial_balance: float = 10_000.0,
    spread_price: float = 0.0,
    slippage_price: float = 0.0,
    commission_per_lot: float = 0.0,
    default_position_mode: PositionMode = PositionMode.NETTING_SINGLE,
    force_close_at_end: bool = True,
    reverse_policy: ReversePolicy = ReversePolicy.IGNORE,
    timeframe: str | None = None,
    analytics: AnalyticsConfig | None = None,
) -> EngineConfig:
    return EngineConfig(
        account=AccountConfig(initial_balance=initial_balance, base_currency="USD", leverage=100.0),
        instrument=InstrumentConfig(
            symbol="XAUUSD",
            lot_size=0.01,
            contract_size=100.0,
            point_size=0.01,
        ),
        spread=SpreadConfig(spread_price=spread_price),
        fee=FeeConfig(
            fee_rate_per_side=0.0,
            commission_per_lot=commission_per_lot,
            commission_mode=CommissionMode.ROUND_TURN,
        ),
        slippage=SlippageConfig(slippage_price=slippage_price),
        risk=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
        ),
        default_position_mode=default_position_mode,
        force_close_at_end=force_close_at_end,
        reverse_policy=reverse_policy,
        timeframe=timeframe,
        analytics=analytics or AnalyticsConfig(),
    )


# ==============================================================================
# 1. END-TO-END R-MULTIPLE: STOP LOSS (-1R), TAKE PROFIT (+2R), STRATEGY EXIT
# ==============================================================================


def test_e2e_r_multiple_tracking_through_stop_loss_take_profit_and_strategy_exit():
    times = pd.date_range("2025-01-01 00:00:00", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "time": times,
            "open":  [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2015.0, 2015.0, 2015.0],
            "high":  [2001.0, 2002.0, 2002.0, 2002.0, 2025.0, 2002.0, 2016.0, 2016.0, 2016.0, 2016.0],
            "low":   [1999.0, 1998.0, 1985.0, 1998.0, 1998.0, 1998.0, 1998.0, 2014.0, 2014.0, 2014.0],
            "close": [2000.0, 2000.0, 1990.0, 2000.0, 2022.0, 2000.0, 2015.0, 2015.0, 2015.0, 2015.0],
            "volume": [1000.0] * 10,
        }
    )
    # Trade 1: Phát tín hiệu tại Bar 0 -> khớp tại Bar 1 open = 2000, SL = 1990 (1R = 10$), chạm SL tại Bar 2 low = 1985 => -1R
    # Trade 2: Phát tín hiệu tại Bar 3 -> khớp tại Bar 4 open = 2000, SL = 1990 (1R = 10$), TP = 2020 (+2R = 20$), chạm TP tại Bar 4 high = 2025 => +2R
    # Trade 3: Phát tín hiệu tại Bar 5 (không SL) -> khớp tại Bar 6 open = 2000, đóng tại Bar 7 open = 2015 => r_multiple = None
    signals = {
        0: _sig(times[0], "XAUUSD", SignalType.LONG, quantity=0.01, stop_loss_price=1990.0, take_profit_price=2050.0),
        3: _sig(times[3], "XAUUSD", SignalType.LONG, quantity=0.01, stop_loss_price=1990.0, take_profit_price=2020.0),
        5: _sig(times[5], "XAUUSD", SignalType.LONG, quantity=0.01),
        6: _sig(times[6], "XAUUSD", SignalType.EXIT_LONG),
    }

    cfg = _base_config(initial_balance=10_000.0)
    engine = BacktestEngine(cfg)
    result = engine.run(df, ScriptedEventStrategy(signals))

    assert len(result.trades) == 3
    t1 = result.trades.iloc[0]
    t2 = result.trades.iloc[1]
    t3 = result.trades.iloc[2]

    assert t1["initial_risk_amount"] == pytest.approx(10.0)
    assert t1["net_pnl"] == pytest.approx(-10.0)
    assert t1["r_multiple"] == pytest.approx(-1.0)

    assert t2["initial_risk_amount"] == pytest.approx(10.0)
    assert t2["net_pnl"] == pytest.approx(20.0)
    assert t2["r_multiple"] == pytest.approx(2.0)

    assert pd.isna(t3["initial_risk_amount"])
    assert pd.isna(t3["r_multiple"])

    # Kiểm tra metrics Group G
    assert result.metrics["valid_r_trades"] == 2
    assert result.metrics["r_coverage_pct"] == pytest.approx(200.0 / 3.0)
    assert result.metrics["average_r"] == pytest.approx(0.5)  # (-1 + 2) / 2
    assert result.metrics["expectancy_r"] == pytest.approx(0.5)
    assert result.metrics["sqn"] is not None


# ==============================================================================
# 2. SCALE-IN + PARTIAL CLOSE + KEEP_OPEN + GATE 03 RECONCILIATION
# ==============================================================================


def test_e2e_scale_in_partial_close_and_keep_open_reconciles_gate_03_and_risk_conservation():
    df = _make_ohlcv_df([2000.0, 2000.0, 2010.0, 2010.0, 2020.0, 2025.0, 2030.0])
    times = df["time"].tolist()
    signals = {
        0: _sig(times[0], "XAUUSD", SignalType.LONG, quantity=0.02, stop_loss_price=1990.0),
        2: _sig(times[2], "XAUUSD", SignalType.LONG, quantity=0.02, stop_loss_price=2000.0),
        4: _sig(times[4], "XAUUSD", SignalType.EXIT_LONG, quantity=0.03),
    }

    cfg = _base_config(
        initial_balance=10_000.0,
        default_position_mode=PositionMode.NETTING_SCALE_IN,
        force_close_at_end=False,
        spread_price=0.10,
        slippage_price=0.02,
        commission_per_lot=7.0,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(df, ScriptedEventStrategy(signals))

    # Vẫn còn 0.01 lot mở ở cuối kỳ
    assert len(result.open_positions) == 1
    assert result.open_positions.iloc[0]["quantity"] == pytest.approx(0.01)
    assert result.open_positions.iloc[0]["remaining_risk_amount"] > 0.0

    # Report Accounting Reconciliation (Group G Gate 03) hợp lệ tuyệt đối
    assert result.accounting_reconciliation is not None
    assert result.accounting_reconciliation.is_valid is True
    assert result.metrics["accounting_reconciled"] is True
    assert result.accounting_reconciliation.difference == pytest.approx(0.0, abs=1e-9)


# ==============================================================================
# 3. BENCHMARK COMPARISON (AUTO BUY & HOLD vs EXTERNAL BENCHMARK)
# ==============================================================================


def test_e2e_auto_buy_and_hold_benchmark_and_external_benchmark_override():
    closes = [2000.0, 2005.0, 2010.0, 2008.0, 2020.0, 2025.0, 2030.0, 2040.0]
    df = _make_ohlcv_df(closes)
    times = df["time"].tolist()
    signals = {
        1: _sig(times[1], "XAUUSD", SignalType.LONG, quantity=0.05, stop_loss_price=1980.0),
        5: _sig(times[5], "XAUUSD", SignalType.EXIT_LONG),
    }

    cfg = _base_config(initial_balance=10_000.0)
    engine = BacktestEngine(cfg)
    res_auto = engine.run(df, ScriptedEventStrategy(signals, warmup=1))

    assert not res_auto.benchmark_equity.empty
    # Warm-up = 1 => effective_trade_start = 1, tại Bar 0 và Bar 1 benchmark_equity == 10000
    assert res_auto.benchmark_equity["equity"].iloc[0] == pytest.approx(10_000.0)
    assert res_auto.benchmark_equity["equity"].iloc[1] == pytest.approx(10_000.0)
    assert res_auto.metrics["benchmark_total_return_pct"] is not None
    assert res_auto.metrics["beta"] is not None
    assert res_auto.metrics["active_return_pct"] is not None

    # Truyền External Benchmark riêng
    ext_bench = pd.DataFrame(
        {
            "time": df["time"],
            "equity": [10_000.0 + i * 50.0 for i in range(len(df))],
        }
    )
    res_ext = engine.run(
        df,
        ScriptedEventStrategy(signals, warmup=1),
        benchmark_data=ext_bench,
    )
    expected_ext_ret = ((10_000.0 + 7 * 50.0) / 10_000.0 - 1.0) * 100.0
    assert res_ext.metrics["benchmark_total_return_pct"] == pytest.approx(
        expected_ext_ret, rel=1e-9
    )


# ==============================================================================
# 4. MULTI-MONTH PERIOD BREAKDOWN, MONTHLY MATRIX & CUSTOM REGIMES
# ==============================================================================


def test_e2e_multi_month_period_stats_monthly_matrix_and_custom_regimes():
    times = pd.date_range("2025-01-15 00:00:00", periods=80, freq="1D", tz="UTC")
    closes = [2000.0 + i * 2.0 for i in range(80)]
    df = pd.DataFrame(
        {
            "time": times,
            "open": closes,
            "high": [c + 5.0 for c in closes],
            "low": [c - 5.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * 80,
        }
    )
    signals = {
        1: _sig(times[1], "XAUUSD", SignalType.LONG, quantity=0.02, stop_loss_price=1980.0),
        10: _sig(times[10], "XAUUSD", SignalType.EXIT_LONG),
        25: _sig(times[25], "XAUUSD", SignalType.LONG, quantity=0.02, stop_loss_price=2020.0),
        40: _sig(times[40], "XAUUSD", SignalType.EXIT_LONG),
    }

    cfg = _base_config(initial_balance=10_000.0, timeframe="D1")
    result = BacktestEngine(cfg).run(df, ScriptedEventStrategy(signals))

    # Dữ liệu trải qua Jan, Feb, Mar, Apr 2025
    assert isinstance(result.period_stats, pd.DataFrame)
    assert len(result.period_stats) >= 3
    assert len(result.monthly_return_matrix) == 1
    assert result.monthly_return_matrix.iloc[0]["year"] == 2025

    q_df = analyze_by_period(
        result.equity,
        result.trades,
        freq="QE",
        ledger_df=result.ledger,
        initial_balance=cfg.initial_balance,
    )
    assert len(q_df) == 2

    # Kiểm tra Custom Regimes trên kết quả BacktestResult
    regimes = [
        ("Jan_Window", times[0], times[15]),
        ("Feb_Window", times[20], times[45]),
    ]
    reg_df = analyze_custom_regimes(
        result.equity,
        result.trades,
        regimes=regimes,
        ledger_df=result.ledger,
        initial_balance=cfg.initial_balance,
    )
    assert len(reg_df) == 2
    assert reg_df["trades"].tolist() == [1, 1]
    assert reg_df.loc[0, "return_pct"] > 0.0
    assert reg_df.loc[1, "return_pct"] > 0.0


# ==============================================================================
# 5. RESULT ARTIFACTS, REPORT PRINTING & REPEATED-RUN DETERMINISM
# ==============================================================================


def test_e2e_backtest_result_contains_all_group_g_tables_and_deterministic_runs(capsys):
    closes = [2000.0, 2010.0, 1995.0, 2020.0, 2005.0, 2035.0, 2025.0, 2050.0]
    df = _make_ohlcv_df(closes)
    times = df["time"].tolist()
    signals = {
        0: _sig(times[0], "XAUUSD", SignalType.LONG, quantity=0.02, stop_loss_price=1980.0),
        3: _sig(times[3], "XAUUSD", SignalType.SHORT, quantity=0.02, stop_loss_price=2040.0),
        5: _sig(times[5], "XAUUSD", SignalType.EXIT_SHORT),
    }

    cfg = _base_config(
        initial_balance=10_000.0,
        reverse_policy=ReversePolicy.TWO_ORDER_REVERSE,
        analytics=AnalyticsConfig(risk_free_rate_annual=0.02, enable_buy_and_hold_benchmark=True),
    )
    engine = BacktestEngine(cfg)
    r1 = engine.run(df, ScriptedEventStrategy(signals))
    r2 = engine.run(df, ScriptedEventStrategy(signals))

    # Đảm bảo tính tất định tuyệt đối giữa 2 lần chạy
    assert r1.metrics == r2.metrics
    pd.testing.assert_frame_equal(r1.returns_series, r2.returns_series)
    pd.testing.assert_frame_equal(r1.drawdown_series, r2.drawdown_series)
    pd.testing.assert_frame_equal(r1.drawdown_episodes, r2.drawdown_episodes)
    pd.testing.assert_frame_equal(r1.monthly_return_matrix, r2.monthly_return_matrix)

    # Kiểm tra calculate_metrics(r1, cfg) và print_report(r1.metrics)
    recalc = calculate_metrics(r1, cfg)
    assert recalc == r1.metrics

    report_str = print_report(r1.metrics, title="INTEGRATION REPORT")
    captured = capsys.readouterr().out
    assert "INTEGRATION REPORT" in report_str
    assert "INTEGRATION REPORT" in captured
