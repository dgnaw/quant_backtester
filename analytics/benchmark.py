from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

from quant_backtester.analytics.drawdown import compute_drawdown_series
from quant_backtester.analytics.performance import (
    annual_to_bar_rate,
    calculate_annualized_volatility,
    calculate_cagr_pct,
    calculate_sharpe_ratio,
)
from quant_backtester.analytics.returns import compute_return_series


BENCHMARK_SERIES_COLUMNS = [
    "time",
    "equity",
    "benchmark_return",
    "cumulative_return",
]

EMPTY_BENCHMARK_METRICS: dict[str, float | None] = {
    "benchmark_total_return_pct": None,
    "benchmark_cagr_pct": None,
    "benchmark_max_drawdown_pct": None,
    "benchmark_sharpe_annual": None,
    "benchmark_volatility_annual": None,
    "active_return_pct": None,
    "beta": None,
    "alpha_annual_pct": None,
    "tracking_error_annual": None,
    "information_ratio": None,
    "up_capture_pct": None,
    "down_capture_pct": None,
}


def build_buy_and_hold_benchmark(
    market_data: pd.DataFrame | None,
    initial_balance: float = 1_000.0,
    effective_trade_start: int = 0,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Tạo chuỗi Benchmark Buy & Hold cho một Symbol cụ thể từ dữ liệu giá thị trường:
    - Với các Bar i < effective_trade_start (vùng warm-up): benchmark_equity = initial_balance.
    - Với các Bar i >= effective_trade_start: benchmark_equity_t = initial_balance * (close_t / close_start).
    """
    if market_data is None or market_data.empty:
        return pd.DataFrame(columns=BENCHMARK_SERIES_COLUMNS)
    if "time" not in market_data.columns or price_col not in market_data.columns:
        raise ValueError(
            f"market_data phải có cột 'time' và '{price_col}' để tạo Buy & Hold Benchmark."
        )
    if initial_balance <= 0.0:
        raise ValueError("initial_balance cho Benchmark phải > 0.")

    df = market_data.copy().reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Timestamp trong market_data của Benchmark phải tăng dần và không trùng lặp.")

    prices = df[price_col].astype(float).to_numpy()
    if np.any(prices <= 0.0) or not np.all(np.isfinite(prices)):
        raise ValueError(f"Cột '{price_col}' chứa giá trị không dương hoặc không hữu hạn.")

    n = len(df)
    start_idx = max(0, min(int(effective_trade_start), n - 1))
    base_price = float(prices[start_idx])

    eq = np.full(n, float(initial_balance), dtype=float)
    for i in range(start_idx, n):
        eq[i] = float(initial_balance) * (float(prices[i]) / base_price)

    ret_df = compute_return_series(
        pd.DataFrame({"time": df["time"], "equity": eq}),
        initial_balance=float(initial_balance),
    )
    return pd.DataFrame(
        {
            "time": ret_df["time"],
            "equity": ret_df["equity"],
            "benchmark_return": ret_df["twr_return"],
            "cumulative_return": ret_df["cumulative_return"],
        },
        columns=BENCHMARK_SERIES_COLUMNS,
    )


def prepare_external_benchmark(
    benchmark_data: pd.DataFrame | pd.Series | None,
    target_times: pd.Series | pd.Index | None = None,
    initial_balance: float = 1_000.0,
) -> pd.DataFrame:
    """
    Chuẩn hóa và căn chỉnh (Align) chuỗi External Benchmark với chuỗi thời gian của chiến lược.
    Hỗ trợ DataFrame có cột 'equity' / 'close' / 'price' / 'benchmark_return' / 'return'
    hoặc Series có DatetimeIndex.
    """
    if benchmark_data is None:
        return pd.DataFrame(columns=BENCHMARK_SERIES_COLUMNS)

    if isinstance(benchmark_data, pd.Series):
        if benchmark_data.empty:
            return pd.DataFrame(columns=BENCHMARK_SERIES_COLUMNS)
        df = pd.DataFrame(
            {"time": pd.to_datetime(benchmark_data.index), "value": benchmark_data.astype(float).to_numpy()}
        )
        val_col = "value"
    elif isinstance(benchmark_data, pd.DataFrame):
        if benchmark_data.empty:
            return pd.DataFrame(columns=BENCHMARK_SERIES_COLUMNS)
        df = benchmark_data.copy()
        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "time"})
            else:
                raise ValueError("External benchmark DataFrame phải có cột 'time' hoặc DatetimeIndex.")
        df["time"] = pd.to_datetime(df["time"])
        val_col = None
        for candidate in ("equity", "close", "price", "benchmark_return", "return"):
            if candidate in df.columns:
                val_col = candidate
                break
        if val_col is None:
            raise ValueError(
                "External benchmark DataFrame phải có một trong các cột: 'equity', 'close', 'price', 'benchmark_return', 'return'."
            )
    else:
        raise TypeError("benchmark_data phải là pd.DataFrame, pd.Series hoặc None.")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("Timestamp của Benchmark phải được sắp xếp tăng dần.")
    if df["time"].duplicated().any():
        raise ValueError("Timestamp của Benchmark không được trùng lặp.")

    if target_times is not None:
        target_dt = pd.DataFrame({"time": pd.to_datetime(pd.Series(list(target_times)))})
        df = pd.merge(target_dt, df, on="time", how="inner")
        if df.empty:
            return pd.DataFrame(columns=BENCHMARK_SERIES_COLUMNS)

    vals = df[val_col].astype(float).to_numpy()
    if val_col in {"benchmark_return", "return"}:
        wealth = np.cumprod(1.0 + vals)
        eq = float(initial_balance) * wealth
        cum_ret = wealth - 1.0
        return pd.DataFrame(
            {
                "time": df["time"],
                "equity": eq,
                "benchmark_return": vals,
                "cumulative_return": cum_ret,
            },
            columns=BENCHMARK_SERIES_COLUMNS,
        )

    if np.any(vals <= 0.0):
        raise ValueError(f"Cột '{val_col}' của Benchmark phải dương (> 0).")
    eq_scaled = float(initial_balance) * (vals / float(vals[0]))
    ret_df = compute_return_series(
        pd.DataFrame({"time": df["time"], "equity": eq_scaled}),
        initial_balance=float(initial_balance),
    )
    return pd.DataFrame(
        {
            "time": ret_df["time"],
            "equity": ret_df["equity"],
            "benchmark_return": ret_df["twr_return"],
            "cumulative_return": ret_df["cumulative_return"],
        },
        columns=BENCHMARK_SERIES_COLUMNS,
    )


def calculate_benchmark_metrics(
    strategy_equity_df: pd.DataFrame | None,
    benchmark_data: pd.DataFrame | pd.Series | None,
    *,
    bars_per_year: int = 96 * 252,
    risk_free_rate_annual: float = 0.0,
    strategy_ledger_df: pd.DataFrame | None = None,
    initial_balance: float | None = None,
) -> dict[str, float | None]:
    """
    Tính bộ chỉ số so sánh Chiến lược với Benchmark trên các mốc thời gian đã căn chỉnh:
    - benchmark_total_return_pct, benchmark_cagr_pct, benchmark_max_drawdown_pct,
      benchmark_sharpe_annual, benchmark_volatility_annual.
    - active_return_pct, beta, alpha_annual_pct, tracking_error_annual,
      information_ratio, up_capture_pct, down_capture_pct.
    Nếu thiếu dữ liệu Benchmark, trả về None cho toàn bộ khóa Benchmark (không điền 0 giả).
    """
    if (
        strategy_equity_df is None
        or strategy_equity_df.empty
        or benchmark_data is None
        or (isinstance(benchmark_data, (pd.DataFrame, pd.Series)) and benchmark_data.empty)
    ):
        return dict(EMPTY_BENCHMARK_METRICS)

    eff_init = (
        float(initial_balance)
        if initial_balance is not None
        else float(strategy_equity_df["equity"].iloc[0])
    )
    strat_ret_df = compute_return_series(
        strategy_equity_df, ledger_df=strategy_ledger_df, initial_balance=eff_init
    )
    strat_ret_df["time"] = pd.to_datetime(strat_ret_df["time"])

    bench_df = prepare_external_benchmark(
        benchmark_data, target_times=strat_ret_df["time"], initial_balance=eff_init
    )
    if len(bench_df) < 2:
        return dict(EMPTY_BENCHMARK_METRICS)

    merged = pd.merge(
        strat_ret_df[["time", "equity", "twr_return", "wealth_index"]],
        bench_df[["time", "equity", "benchmark_return"]],
        on="time",
        how="inner",
        suffixes=("_strat", "_bench"),
    )
    if len(merged) < 2:
        return dict(EMPTY_BENCHMARK_METRICS)

    # Chuẩn hóa lại lợi suất trên lưới thời gian đã căn chỉnh (từ quan sát 1..M-1)
    w_s = merged["wealth_index"].astype(float).to_numpy()
    eq_b = merged["equity_bench"].astype(float).to_numpy()

    r_s = (w_s[1:] / w_s[:-1]) - 1.0
    r_b = (eq_b[1:] / eq_b[:-1]) - 1.0
    m_bars = len(r_s)
    if m_bars < 1:
        return dict(EMPTY_BENCHMARK_METRICS)

    strat_total_ret_pct = float(((w_s[-1] / w_s[0]) - 1.0) * 100.0)
    bench_total_ret_pct = float(((eq_b[-1] / eq_b[0]) - 1.0) * 100.0)
    active_ret_pct = float(strat_total_ret_pct - bench_total_ret_pct)

    bench_cagr = calculate_cagr_pct(
        eq_b[-1] / eq_b[0], n_return_bars=m_bars, bars_per_year=bars_per_year
    )
    bench_dd_df = compute_drawdown_series(
        pd.DataFrame({"time": merged["time"], "equity": eq_b}),
        initial_balance=float(eq_b[0]),
    )
    bench_max_dd_pct = float(bench_dd_df["drawdown_pct"].min()) if not bench_dd_df.empty else 0.0
    bench_sharpe = calculate_sharpe_ratio(
        r_b, bars_per_year=bars_per_year, risk_free_rate_annual=risk_free_rate_annual
    )
    bench_vol = calculate_annualized_volatility(r_b, bars_per_year=bars_per_year)

    rf_bar = annual_to_bar_rate(risk_free_rate_annual, bars_per_year)
    beta: float | None = None
    alpha_annual_pct: float | None = None
    tracking_error: float | None = None
    info_ratio: float | None = None

    if m_bars >= 2:
        var_b = float(np.var(r_b, ddof=1))
        if var_b > 1e-16:
            cov_sb = float(np.cov(r_s, r_b, ddof=1)[0, 1])
            beta = float(cov_sb / var_b)
        else:
            beta = 0.0

        mean_ex_s = float(np.mean(r_s - rf_bar))
        mean_ex_b = float(np.mean(r_b - rf_bar))
        alpha_annual_pct = float(
            ((mean_ex_s - beta * mean_ex_b) * float(bars_per_year)) * 100.0
        )

        diff = r_s - r_b
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        tracking_error = float(std_diff * math.sqrt(float(bars_per_year)))
        if std_diff <= 1e-12:
            if abs(mean_diff) <= 1e-12:
                info_ratio = 0.0
            else:
                info_ratio = float("inf") if mean_diff > 0.0 else float("-inf")
        else:
            info_ratio = float((mean_diff / std_diff) * math.sqrt(float(bars_per_year)))

    # Up / Down Capture (trung bình nhân trên tập kỳ Benchmark tăng / giảm)
    up_mask = r_b > 1e-15
    down_mask = r_b < -1e-15
    up_capture_pct: float | None = None
    down_capture_pct: float | None = None

    if np.any(up_mask):
        n_up = int(np.sum(up_mask))
        prod_s_up = float(np.prod(1.0 + r_s[up_mask]))
        prod_b_up = float(np.prod(1.0 + r_b[up_mask]))
        if prod_s_up > 0.0 and prod_b_up > 0.0:
            geo_s_up = (prod_s_up ** (1.0 / n_up)) - 1.0
            geo_b_up = (prod_b_up ** (1.0 / n_up)) - 1.0
            if abs(geo_b_up) > 1e-15:
                up_capture_pct = float((geo_s_up / geo_b_up) * 100.0)

    if np.any(down_mask):
        n_dn = int(np.sum(down_mask))
        prod_s_dn = float(np.prod(1.0 + r_s[down_mask]))
        prod_b_dn = float(np.prod(1.0 + r_b[down_mask]))
        if prod_s_dn > 0.0 and prod_b_dn > 0.0:
            geo_s_dn = (prod_s_dn ** (1.0 / n_dn)) - 1.0
            geo_b_dn = (prod_b_dn ** (1.0 / n_dn)) - 1.0
            if abs(geo_b_dn) > 1e-15:
                down_capture_pct = float((geo_s_dn / geo_b_dn) * 100.0)

    return {
        "benchmark_total_return_pct": bench_total_ret_pct,
        "benchmark_cagr_pct": bench_cagr,
        "benchmark_max_drawdown_pct": bench_max_dd_pct,
        "benchmark_sharpe_annual": bench_sharpe,
        "benchmark_volatility_annual": bench_vol,
        "active_return_pct": active_ret_pct,
        "beta": beta,
        "alpha_annual_pct": alpha_annual_pct,
        "tracking_error_annual": tracking_error,
        "information_ratio": info_ratio,
        "up_capture_pct": up_capture_pct,
        "down_capture_pct": down_capture_pct,
    }
