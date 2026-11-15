from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
import pandas as pd

from quant_backtester.analytics.accounting import (
    LedgerCashFlowSummary,
    ReportAccountingReconciliation,
    reconcile_report_accounting,
    summarize_ledger_cash_flows,
)
from quant_backtester.analytics.annualization import resolve_bars_per_year
from quant_backtester.analytics.benchmark import (
    BENCHMARK_SERIES_COLUMNS,
    calculate_benchmark_metrics,
    prepare_external_benchmark,
)
from quant_backtester.analytics.drawdown import (
    compute_drawdown_series,
    extract_drawdown_episodes,
    summarize_drawdown,
)
from quant_backtester.analytics.performance import (
    calculate_annualized_volatility,
    calculate_cagr_pct,
    calculate_calmar_ratio,
    calculate_downside_deviation,
    calculate_return_distribution_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    pd_valid,
)
from quant_backtester.analytics.period_analysis import (
    analyze_by_period,
    build_monthly_return_matrix,
)
from quant_backtester.analytics.returns import (
    compute_return_series,
    extract_active_period_returns,
)
from quant_backtester.analytics.trade_analysis import analyze_trades


@dataclass(frozen=True)
class PerformanceSummary:
    """Bản tóm tắt các chỉ số hiệu suất định lượng chuẩn hóa."""

    metrics: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.metrics[key]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.metrics)


@dataclass
class AnalyticsResult:
    """
    Đối tượng đóng gói toàn bộ kết quả phân tích của Group G:
    - metrics: dict chỉ số tổng hợp (giữ đủ 11 khóa cũ và các khóa mở rộng).
    - accounting_summary: tổng hợp dòng tiền Ledger.
    - reconciliation: báo cáo đối soát Gate 03.
    - returns_series: bảng chuỗi lợi suất (Simple, Log, TWR, Wealth Index).
    - drawdown_series: bảng chuỗi Drawdown & Underwater.
    - drawdown_episodes: bảng các đợt sụt giảm (Peak -> Valley -> Recovery).
    - monthly_stats / quarterly_stats / yearly_stats: bảng hiệu suất theo giai đoạn.
    - monthly_return_matrix: bảng Pivot Year x [Jan..Dec, YTD].
    - benchmark_series: chuỗi Benchmark đã căn chỉnh.
    """

    metrics: dict[str, Any]
    accounting_summary: LedgerCashFlowSummary
    reconciliation: ReportAccountingReconciliation
    returns_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_episodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    yearly_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_return_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_series: pd.DataFrame = field(default_factory=pd.DataFrame)


def _extract_inputs(
    result: Any,
    config: Any = None,
    benchmark_data: pd.DataFrame | pd.Series | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    Any,
    pd.DataFrame | pd.Series | None,
]:
    if isinstance(result, dict):
        equity_df = result.get("equity")
        trades_df = result.get("trades")
        ledger_df = result.get("ledger")
        open_positions_df = result.get("open_positions")
        eff_cfg = config if config is not None else result.get("config")
        eff_bench = (
            benchmark_data
            if benchmark_data is not None
            else result.get("benchmark_equity")
        )
    else:
        equity_df = getattr(result, "equity", None)
        trades_df = getattr(result, "trades", None)
        ledger_df = getattr(result, "ledger", None)
        open_positions_df = getattr(result, "open_positions", None)
        eff_cfg = config if config is not None else getattr(result, "config", None)
        eff_bench = (
            benchmark_data
            if benchmark_data is not None
            else getattr(result, "benchmark_equity", None)
        )

    if equity_df is None or not isinstance(equity_df, pd.DataFrame) or equity_df.empty:
        raise ValueError("Equity curve rỗng.")
    if "equity" not in equity_df.columns:
        raise ValueError("equity_df phải có cột 'equity'.")

    if trades_df is None:
        trades_df = pd.DataFrame()
    return equity_df, trades_df, ledger_df, open_positions_df, eff_cfg, eff_bench


def build_analytics_result(
    result: Any,
    config: Any = None,
    benchmark_data: pd.DataFrame | pd.Series | None = None,
) -> AnalyticsResult:
    """
    Thực hiện toàn bộ quy trình phân tích Group G (Pure Function, không sửa dữ liệu đầu vào)
    và trả về đối tượng AnalyticsResult đầy đủ.
    """
    equity_df, trades_df, ledger_df, open_pos_df, eff_cfg, eff_bench = _extract_inputs(
        result, config=config, benchmark_data=benchmark_data
    )

    eq_clean = equity_df.copy().reset_index(drop=True)
    if "time" not in eq_clean.columns:
        eq_clean["time"] = pd.date_range("2000-01-01", periods=len(eq_clean), freq="15min")

    init_bal = (
        float(getattr(eff_cfg, "initial_balance", eq_clean["equity"].iloc[0]))
        if eff_cfg is not None
        else float(eq_clean["equity"].iloc[0])
    )
    bars_per_year = resolve_bars_per_year(config=eff_cfg, time_series=eq_clean["time"])

    rf_annual = 0.0
    if eff_cfg is not None:
        raw_rf = getattr(eff_cfg, "risk_free_rate_annual", None)
        if raw_rf is None and hasattr(eff_cfg, "analytics"):
            raw_rf = getattr(eff_cfg.analytics, "risk_free_rate_annual", 0.0)
        if raw_rf is not None:
            rf_annual = float(raw_rf)

    acc_summary = summarize_ledger_cash_flows(ledger_df, initial_balance=init_bal)
    reconciliation = reconcile_report_accounting(
        equity_df=eq_clean,
        trades_df=trades_df,
        ledger_df=ledger_df,
        open_positions_df=open_pos_df,
        initial_balance=init_bal,
    )

    returns_df = compute_return_series(
        eq_clean, ledger_df=ledger_df, initial_balance=init_bal
    )
    active_returns = extract_active_period_returns(returns_df, initial_balance=init_bal)

    drawdown_df = compute_drawdown_series(
        eq_clean, ledger_df=ledger_df, initial_balance=init_bal
    )
    episodes_df = extract_drawdown_episodes(drawdown_df)
    dd_summary = summarize_drawdown(
        drawdown_df,
        net_trading_profit=reconciliation.actual_value,
        episodes_df=episodes_df,
    )

    final_equity = float(eq_clean["equity"].iloc[-1])
    final_balance = (
        float(eq_clean["balance"].iloc[-1])
        if "balance" in eq_clean.columns
        else final_equity
    )

    # TWR Total Return (đã loại bỏ ảnh hưởng của Deposit/Withdrawal)
    final_wealth = float(returns_df["wealth_index"].iloc[-1])
    twr_total_return_pct = float((final_wealth - 1.0) * 100.0)
    # Khi không có dòng tiền ngoại sinh, total_return_pct == twr_total_return_pct == (final_equity / init_bal - 1) * 100
    if abs(acc_summary.net_external_cash_flow) <= 1e-12:
        total_return_pct = float((final_equity / init_bal - 1.0) * 100.0)
    else:
        total_return_pct = twr_total_return_pct

    n_ret_bars = len(active_returns)
    cagr_pct = calculate_cagr_pct(
        final_wealth, n_return_bars=n_ret_bars, bars_per_year=bars_per_year
    )
    vol_annual = calculate_annualized_volatility(active_returns, bars_per_year=bars_per_year)
    downside_dev_annual = calculate_downside_deviation(
        active_returns,
        bars_per_year=bars_per_year,
        risk_free_rate_annual=rf_annual,
        annualize=True,
    )

    raw_sharpe = calculate_sharpe_ratio(
        active_returns,
        bars_per_year=bars_per_year,
        risk_free_rate_annual=rf_annual,
    )
    if raw_sharpe is None:
        sharpe_annual = 0.0
        sharpe_is_valid = False
    else:
        sharpe_annual = float(raw_sharpe)
        sharpe_is_valid = True

    sortino_annual = calculate_sortino_ratio(
        active_returns,
        bars_per_year=bars_per_year,
        risk_free_rate_annual=rf_annual,
    )
    calmar_ratio = calculate_calmar_ratio(cagr_pct, dd_summary["max_drawdown_pct"])
    dist_metrics = calculate_return_distribution_metrics(active_returns)
    trade_stats = analyze_trades(trades_df)

    bench_metrics = calculate_benchmark_metrics(
        eq_clean,
        eff_bench,
        bars_per_year=bars_per_year,
        risk_free_rate_annual=rf_annual,
        strategy_ledger_df=ledger_df,
        initial_balance=init_bal,
    )
    if eff_bench is not None and not (
        isinstance(eff_bench, (pd.DataFrame, pd.Series)) and eff_bench.empty
    ):
        bench_series = prepare_external_benchmark(
            eff_bench, target_times=eq_clean["time"], initial_balance=init_bal
        )
    else:
        bench_series = pd.DataFrame(columns=BENCHMARK_SERIES_COLUMNS)

    monthly_stats = analyze_by_period(
        eq_clean, trades_df, freq="ME", ledger_df=ledger_df, initial_balance=init_bal
    )
    quarterly_stats = analyze_by_period(
        eq_clean, trades_df, freq="QE", ledger_df=ledger_df, initial_balance=init_bal
    )
    yearly_stats = analyze_by_period(
        eq_clean, trades_df, freq="YE", ledger_df=ledger_df, initial_balance=init_bal
    )
    monthly_matrix = build_monthly_return_matrix(
        eq_clean, ledger_df=ledger_df, initial_balance=init_bal
    )

    metrics: dict[str, Any] = {
        # 11 khóa gốc (Backward-Compatible Contract)
        "initial_balance": init_bal,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": float(dd_summary["max_drawdown_pct"]),
        "sharpe_annual": sharpe_annual,
        "total_trades": int(trade_stats["total_trades"]),
        "wins": int(trade_stats["wins"]),
        "losses": int(trade_stats["losses"]),
        "win_rate_pct": float(trade_stats["win_rate_pct"]),
        "profit_factor": float(trade_stats["profit_factor"]),
        "expectancy_cash_per_trade": float(trade_stats["expectancy_cash_per_trade"]),
        # Accounting & Cash Flow
        "final_balance": final_balance,
        "total_deposits": acc_summary.total_deposits,
        "total_withdrawals": acc_summary.total_withdrawals,
        "net_external_cash_flow": acc_summary.net_external_cash_flow,
        "net_trading_profit": reconciliation.actual_value,
        "accounting_reconciled": reconciliation.is_valid,
        # Return & Risk-Adjusted Ratios
        "twr_total_return_pct": twr_total_return_pct,
        "cagr_pct": cagr_pct,
        "bars_per_year": int(bars_per_year),
        "risk_free_rate_annual": rf_annual,
        "volatility_annual": vol_annual,
        "downside_deviation_annual": downside_dev_annual,
        "sharpe_is_valid": sharpe_is_valid,
        "sortino_annual": sortino_annual,
        "calmar_ratio": calmar_ratio,
        **dist_metrics,
        # Drawdown Extended Metrics
        "max_drawdown_depth_pct": dd_summary["max_drawdown_depth_pct"],
        "max_drawdown_amount": dd_summary["max_drawdown_amount"],
        "avg_drawdown_pct": dd_summary["avg_drawdown_pct"],
        "avg_episode_drawdown_pct": dd_summary["avg_episode_drawdown_pct"],
        "max_drawdown_duration_bars": dd_summary["max_drawdown_duration_bars"],
        "max_drawdown_duration_time": dd_summary["max_drawdown_duration_time"],
        "ulcer_index": dd_summary["ulcer_index"],
        "recovery_factor": dd_summary["recovery_factor"],
        "drawdown_episodes_count": dd_summary["drawdown_episodes_count"],
        # Trade & R-Multiple Extended Metrics
        **trade_stats,
        # Benchmark Metrics
        **bench_metrics,
    }

    return AnalyticsResult(
        metrics=metrics,
        accounting_summary=acc_summary,
        reconciliation=reconciliation,
        returns_series=returns_df,
        drawdown_series=drawdown_df,
        drawdown_episodes=episodes_df,
        monthly_stats=monthly_stats,
        quarterly_stats=quarterly_stats,
        yearly_stats=yearly_stats,
        monthly_return_matrix=monthly_matrix,
        benchmark_series=bench_series,
    )


def calculate_metrics(
    result: Any,
    config: Any = None,
    benchmark_data: pd.DataFrame | pd.Series | None = None,
) -> dict[str, Any]:
    """
    Tính toán từ điển chỉ số hiệu suất (giữ nguyên chữ ký và 11 khóa gốc tương thích ngược,
    đồng thời bổ sung toàn bộ chỉ số Group G).
    """
    analytics_res = build_analytics_result(
        result, config=config, benchmark_data=benchmark_data
    )
    return analytics_res.metrics


def _fmt_val(val: Any) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return f"{val:,d}"
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "+inf" if val > 0 else "-inf"
        return f"{val:,.4f}"
    return str(val)


def print_report(metrics: dict[str, Any], title: str = "BACKTEST REPORT") -> str:
    lines = ["\n" + "=" * 60, title, "=" * 60]
    report_fields = [
        ("initial_balance", "Initial balance"),
        ("final_equity", "Final equity"),
        ("total_return_pct", "Total return (%)"),
        ("cagr_pct", "CAGR (%)"),
        ("max_drawdown_pct", "Max drawdown (%)"),
        ("max_drawdown_amount", "Max drawdown ($)"),
        ("sharpe_annual", "Annualized Sharpe"),
        ("sortino_annual", "Annualized Sortino"),
        ("calmar_ratio", "Calmar ratio"),
        ("recovery_factor", "Recovery factor"),
        ("total_trades", "Total trades"),
        ("win_rate_pct", "Win rate (%)"),
        ("profit_factor", "Profit factor"),
        ("expectancy_cash_per_trade", "Expectancy / trade"),
        ("average_r", "Average R"),
        ("sqn", "SQN (R-based)"),
    ]
    for key, label in report_fields:
        if key in metrics:
            lines.append(f"{label:<25}: {_fmt_val(metrics[key])}")
    text = "\n".join(lines)
    print(text)
    return text
