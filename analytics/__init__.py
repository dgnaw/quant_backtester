"""
Group G: Accounting & Performance Analytics Package.
Re-exports legacy API (`calculate_metrics`, `print_report`, `pd_valid`) and all Group G modules.
"""

from __future__ import annotations

from .accounting import (
    LedgerCashFlowSummary,
    ReportAccountingReconciliation,
    map_cash_flows_to_bars,
    reconcile_report_accounting,
    summarize_ledger_cash_flows,
)
from .annualization import (
    DEFAULT_FOREX_BARS_PER_YEAR,
    TIMEFRAME_MINUTES_MAP,
    bars_per_year_from_timeframe,
    infer_bars_per_year_from_timestamps,
    resolve_bars_per_year,
)
from .benchmark import (
    BENCHMARK_SERIES_COLUMNS,
    EMPTY_BENCHMARK_METRICS,
    build_buy_and_hold_benchmark,
    calculate_benchmark_metrics,
    prepare_external_benchmark,
)
from .drawdown import (
    DRAWDOWN_EPISODE_COLUMNS,
    DRAWDOWN_SERIES_COLUMNS,
    compute_drawdown_series,
    extract_drawdown_episodes,
    summarize_drawdown,
)
from .performance import (
    annual_to_bar_rate,
    calculate_annualized_volatility,
    calculate_cagr_pct,
    calculate_calmar_ratio,
    calculate_downside_deviation,
    calculate_return_distribution_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    pd_valid,
)
from .period_analysis import (
    MONTHLY_MATRIX_COLUMNS,
    PERIOD_STATS_COLUMNS,
    REGIME_STATS_COLUMNS,
    analyze_by_period,
    analyze_custom_regimes,
    build_monthly_return_matrix,
)
from .report import (
    AnalyticsResult,
    PerformanceSummary,
    build_analytics_result,
    calculate_metrics,
    print_report,
)
from .returns import (
    RETURN_SERIES_COLUMNS,
    compute_return_series,
    extract_active_period_returns,
)
from .trade_analysis import analyze_trades

__all__ = [
    "AnalyticsResult",
    "BENCHMARK_SERIES_COLUMNS",
    "DEFAULT_FOREX_BARS_PER_YEAR",
    "DRAWDOWN_EPISODE_COLUMNS",
    "DRAWDOWN_SERIES_COLUMNS",
    "EMPTY_BENCHMARK_METRICS",
    "LedgerCashFlowSummary",
    "MONTHLY_MATRIX_COLUMNS",
    "PERIOD_STATS_COLUMNS",
    "PerformanceSummary",
    "REGIME_STATS_COLUMNS",
    "RETURN_SERIES_COLUMNS",
    "ReportAccountingReconciliation",
    "TIMEFRAME_MINUTES_MAP",
    "analyze_by_period",
    "analyze_custom_regimes",
    "analyze_trades",
    "annual_to_bar_rate",
    "bars_per_year_from_timeframe",
    "build_analytics_result",
    "build_buy_and_hold_benchmark",
    "build_monthly_return_matrix",
    "calculate_annualized_volatility",
    "calculate_benchmark_metrics",
    "calculate_cagr_pct",
    "calculate_calmar_ratio",
    "calculate_downside_deviation",
    "calculate_metrics",
    "calculate_return_distribution_metrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "compute_drawdown_series",
    "compute_return_series",
    "extract_active_period_returns",
    "extract_drawdown_episodes",
    "infer_bars_per_year_from_timestamps",
    "map_cash_flows_to_bars",
    "pd_valid",
    "prepare_external_benchmark",
    "print_report",
    "reconcile_report_accounting",
    "resolve_bars_per_year",
    "summarize_drawdown",
    "summarize_ledger_cash_flows",
]
