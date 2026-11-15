from __future__ import annotations

import math
import numpy as np
import pandas as pd

from quant_backtester.analytics.accounting import (
    map_cash_flows_to_bars,
    summarize_ledger_cash_flows,
)


RETURN_SERIES_COLUMNS = [
    "time",
    "equity",
    "external_cash_flow",
    "simple_return",
    "log_return",
    "twr_return",
    "wealth_index",
    "cumulative_return",
]


def compute_return_series(
    equity_df: pd.DataFrame | None,
    ledger_df: pd.DataFrame | None = None,
    initial_balance: float | None = None,
) -> pd.DataFrame:
    """
    Tính chuỗi lợi suất chuẩn hóa từ Equity Curve và Ledger:
    - simple_return: equity_t / equity_{t-1} - 1
    - log_return: ln(equity_t / equity_{t-1}) khi không có dòng tiền, hoặc ln(1 + twr_return_t)
    - twr_return: (equity_t - CF_t) / equity_{t-1} - 1 (Time-Weighted Return đã khử Deposit/Withdrawal)
    - wealth_index: Khởi tạo từ 1.0, nhân tích lũy (1 + twr_return_t)
    - cumulative_return: wealth_index_t - 1.0
    """
    if equity_df is None or equity_df.empty:
        return pd.DataFrame(columns=RETURN_SERIES_COLUMNS)

    if "time" not in equity_df.columns or "equity" not in equity_df.columns:
        raise ValueError(
            f"equity_df phải có cột 'time' và 'equity'. Hiện có: {list(equity_df.columns)}"
        )

    df = equity_df.copy().reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    if not times.is_monotonic_increasing:
        raise ValueError("Timestamp trong equity_df phải được sắp xếp tăng dần (monotonic increasing).")
    if times.duplicated().any():
        raise ValueError("Timestamp trong equity_df không được trùng lặp.")

    equities = df["equity"].astype(float).to_numpy()
    if np.isnan(equities).any() or not np.all(np.isfinite(equities)):
        raise ValueError("Cột 'equity' trong equity_df chứa giá trị NaN hoặc vô cực.")

    n = len(df)
    eff_init_bal = float(initial_balance) if initial_balance is not None else float(equities[0])
    if eff_init_bal <= 0.0:
        raise ValueError("initial_balance / equity khởi tạo phải lớn hơn 0.")

    cf_summary = summarize_ledger_cash_flows(ledger_df, initial_balance=eff_init_bal)
    cf_per_bar = map_cash_flows_to_bars(times, cf_summary.cash_flows_by_time)

    simple_ret = np.zeros(n, dtype=float)
    log_ret = np.zeros(n, dtype=float)
    twr_ret = np.zeros(n, dtype=float)
    wealth = np.ones(n, dtype=float)
    cum_ret = np.zeros(n, dtype=float)

    # Bar 0
    eq0 = equities[0]
    cf0 = cf_per_bar[0]
    if initial_balance is not None and not math.isclose(
        eq0 - cf0, eff_init_bal, rel_tol=1e-12, abs_tol=1e-12
    ):
        s0 = (eq0 / eff_init_bal) - 1.0
        t0 = ((eq0 - cf0) / eff_init_bal) - 1.0
        simple_ret[0] = s0
        twr_ret[0] = t0
        log_ret[0] = math.log(1.0 + t0) if (1.0 + t0) > 0.0 else float("nan")
        wealth[0] = max(0.0, 1.0 + t0) if (1.0 + t0) > 0.0 else 0.0
        cum_ret[0] = wealth[0] - 1.0
    else:
        simple_ret[0] = 0.0
        twr_ret[0] = 0.0
        log_ret[0] = 0.0 if eq0 > 0.0 else float("nan")
        wealth[0] = 1.0
        cum_ret[0] = 0.0

    # Bars 1 .. n-1
    for t in range(1, n):
        prev_eq = equities[t - 1]
        curr_eq = equities[t]
        cf_t = cf_per_bar[t]
        if prev_eq <= 0.0:
            simple_ret[t] = float("nan")
            twr_ret[t] = float("nan")
            log_ret[t] = float("nan")
            wealth[t] = 0.0
            cum_ret[t] = -1.0
        else:
            s_t = (curr_eq / prev_eq) - 1.0
            t_t = ((curr_eq - cf_t) / prev_eq) - 1.0
            simple_ret[t] = s_t
            twr_ret[t] = t_t
            growth_t = 1.0 + t_t
            if growth_t > 0.0 and curr_eq > 0.0:
                log_ret[t] = math.log(growth_t)
                wealth[t] = wealth[t - 1] * growth_t
            else:
                log_ret[t] = float("nan")
                wealth[t] = 0.0
            cum_ret[t] = wealth[t] - 1.0

    return pd.DataFrame(
        {
            "time": df["time"],
            "equity": equities,
            "external_cash_flow": cf_per_bar,
            "simple_return": simple_ret,
            "log_return": log_ret,
            "twr_return": twr_ret,
            "wealth_index": wealth,
            "cumulative_return": cum_ret,
        },
        columns=RETURN_SERIES_COLUMNS,
    )


def extract_active_period_returns(
    return_series_df: pd.DataFrame,
    initial_balance: float | None = None,
) -> pd.Series:
    """
    Trích xuất chuỗi lợi suất TWR từng kỳ để tính Volatility / Sharpe / Sortino:
    - Nếu Bar 0 là mốc khởi tạo tĩnh (equity_0 - CF_0 == initial_balance), chuỗi kỳ quan sát là từ Bar 1..N-1
      (khớp chính xác độ dài N-1 của pct_change().dropna()).
    - Nếu Bar 0 đã có biến động so với initial_balance được truyền tường minh, lấy từ Bar 0..N-1.
    """
    if return_series_df is None or return_series_df.empty:
        return pd.Series(dtype=float)
    n = len(return_series_df)
    if n == 1:
        r0 = float(return_series_df["twr_return"].iloc[0])
        if initial_balance is not None and abs(r0) > 1e-15:
            return pd.Series([r0], dtype=float)
        return pd.Series(dtype=float)

    eq0 = float(return_series_df["equity"].iloc[0])
    cf0 = float(return_series_df["external_cash_flow"].iloc[0])
    if initial_balance is not None and not math.isclose(
        eq0 - cf0, float(initial_balance), rel_tol=1e-12, abs_tol=1e-12
    ):
        series = return_series_df["twr_return"].astype(float)
    else:
        series = return_series_df["twr_return"].iloc[1:].astype(float)
    return series.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
