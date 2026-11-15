from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from quant_backtester.analytics.returns import compute_return_series
from quant_backtester.analytics.trade_analysis import analyze_trades


PERIOD_STATS_COLUMNS = [
    "period",
    "start_time",
    "end_time",
    "start_equity",
    "end_equity",
    "return_pct",
    "max_drawdown_pct",
    "trades",
    "wins",
    "losses",
    "win_rate_pct",
    "net_pnl",
    "profit_factor",
]

REGIME_STATS_COLUMNS = [
    "regime",
    "start_time",
    "end_time",
    "start_equity",
    "end_equity",
    "return_pct",
    "max_drawdown_pct",
    "trades",
    "wins",
    "losses",
    "win_rate_pct",
    "net_pnl",
    "profit_factor",
]

MONTHLY_MATRIX_COLUMNS = [
    "year",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "YTD",
]

_MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _normalize_period_code(freq: str) -> str:
    f = str(freq).strip().upper()
    if f in {"ME", "M", "MONTH", "MONTHLY"}:
        return "M"
    if f in {"QE", "Q", "QUARTER", "QUARTERLY"}:
        return "Q"
    if f in {"YE", "Y", "A", "YEAR", "YEARLY"}:
        return "Y"
    raise ValueError(
        f"freq '{freq}' không được hỗ trợ. Hỗ trợ: 'ME'/'M' (Monthly), 'QE'/'Q' (Quarterly), 'YE'/'Y' (Yearly)."
    )


def _format_period_label(p: pd.Period, period_code: str) -> str:
    if period_code == "M":
        return f"{p.year:04d}-{p.month:02d}"
    if period_code == "Q":
        return f"{p.year:04d}-Q{p.quarter}"
    return f"{p.year:04d}"


def _compute_local_drawdown_pct(wealth_values: np.ndarray, anchor_wealth: float) -> float:
    if len(wealth_values) == 0:
        return 0.0
    combined = np.concatenate([[float(anchor_wealth)], wealth_values.astype(float)])
    peaks = np.maximum.accumulate(combined)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0.0, (combined / peaks - 1.0) * 100.0, 0.0)
    min_dd = float(np.min(dd))
    return 0.0 if abs(min_dd) <= 1e-12 else min(0.0, min_dd)


def analyze_by_period(
    equity_df: pd.DataFrame | None,
    trades_df: pd.DataFrame | None = None,
    freq: str = "ME",
    ledger_df: pd.DataFrame | None = None,
    initial_balance: float | None = None,
) -> pd.DataFrame:
    """
    Phân tích hiệu suất và thống kê giao dịch theo giai đoạn (Monthly, Quarterly, Yearly):
    - return_pct là lợi suất TWR ghép lãi trong kỳ (đã loại bỏ ảnh hưởng của Deposit/Withdrawal).
    - max_drawdown_pct tính trên Wealth Index nội kỳ (neo từ mức Wealth cuối kỳ liền trước).
    - Giao dịch được phân bổ vào kỳ theo exit_time.
    """
    period_code = _normalize_period_code(freq)
    if equity_df is None or equity_df.empty:
        return pd.DataFrame(columns=PERIOD_STATS_COLUMNS)

    eff_init = (
        float(initial_balance)
        if initial_balance is not None
        else float(equity_df["equity"].iloc[0])
    )
    ret_df = compute_return_series(
        equity_df, ledger_df=ledger_df, initial_balance=eff_init
    )
    ret_df["time_dt"] = pd.to_datetime(ret_df["time"])
    # Loại bỏ timezone khi chuyển sang Period để tránh cảnh báo của pandas
    time_naive = (
        ret_df["time_dt"].dt.tz_localize(None)
        if ret_df["time_dt"].dt.tz is not None
        else ret_df["time_dt"]
    )
    ret_df["_period"] = time_naive.dt.to_period(period_code)

    closed_trades = pd.DataFrame()
    if trades_df is not None and not trades_df.empty and "exit_time" in trades_df.columns:
        closed_trades = trades_df[trades_df["exit_time"].notna()].copy()
        if not closed_trades.empty:
            exit_dt = pd.to_datetime(closed_trades["exit_time"])
            exit_naive = exit_dt.dt.tz_localize(None) if exit_dt.dt.tz is not None else exit_dt
            closed_trades["_period"] = exit_naive.dt.to_period(period_code)

    rows: list[dict[str, Any]] = []
    prev_end_wealth = 1.0
    prev_end_equity = eff_init

    for p_obj, grp in ret_df.groupby("_period", sort=True):
        label = _format_period_label(p_obj, period_code)
        start_time = grp["time"].iloc[0]
        end_time = grp["time"].iloc[-1]
        start_eq = float(prev_end_equity) if rows else float(grp["equity"].iloc[0])
        end_eq = float(grp["equity"].iloc[-1])

        twr_arr = grp["twr_return"].to_numpy(dtype=float)
        wealth_arr = grp["wealth_index"].to_numpy(dtype=float)

        growth = float(np.prod(1.0 + twr_arr))
        period_ret_pct = (growth - 1.0) * 100.0
        if abs(period_ret_pct) <= 1e-12:
            period_ret_pct = 0.0

        local_max_dd = _compute_local_drawdown_pct(wealth_arr, prev_end_wealth)

        if not closed_trades.empty:
            p_trades = closed_trades[closed_trades["_period"] == p_obj]
        else:
            p_trades = pd.DataFrame()

        t_stats = analyze_trades(p_trades)
        rows.append(
            {
                "period": label,
                "start_time": start_time,
                "end_time": end_time,
                "start_equity": start_eq,
                "end_equity": end_eq,
                "return_pct": float(period_ret_pct),
                "max_drawdown_pct": float(local_max_dd),
                "trades": int(t_stats["total_trades"]),
                "wins": int(t_stats["wins"]),
                "losses": int(t_stats["losses"]),
                "win_rate_pct": float(t_stats["win_rate_pct"]),
                "net_pnl": float(t_stats["total_net_pnl"]),
                "profit_factor": float(t_stats["profit_factor"]),
            }
        )
        prev_end_wealth = float(wealth_arr[-1])
        prev_end_equity = end_eq

    if not rows:
        return pd.DataFrame(columns=PERIOD_STATS_COLUMNS)
    return pd.DataFrame(rows, columns=PERIOD_STATS_COLUMNS)


def build_monthly_return_matrix(
    equity_df: pd.DataFrame | None,
    ledger_df: pd.DataFrame | None = None,
    initial_balance: float | None = None,
) -> pd.DataFrame:
    """
    Tạo bảng Monthly Return Matrix (Year x [Jan..Dec, YTD]).
    YTD được tính bằng phép nhân ghép lãi các tháng trong năm:
        YTD = (product(1 + monthly_return) - 1) * 100
    """
    if equity_df is None or equity_df.empty:
        return pd.DataFrame(columns=MONTHLY_MATRIX_COLUMNS)

    monthly_df = analyze_by_period(
        equity_df,
        trades_df=None,
        freq="ME",
        ledger_df=ledger_df,
        initial_balance=initial_balance,
    )
    if monthly_df.empty:
        return pd.DataFrame(columns=MONTHLY_MATRIX_COLUMNS)

    by_year: dict[int, dict[int, float]] = {}
    for _, row in monthly_df.iterrows():
        p_str = str(row["period"])
        year_str, month_str = p_str.split("-")
        yr = int(year_str)
        mo = int(month_str)
        by_year.setdefault(yr, {})[mo] = float(row["return_pct"])

    matrix_rows: list[dict[str, Any]] = []
    for yr in sorted(by_year.keys()):
        month_map = by_year[yr]
        r_dict: dict[str, Any] = {"year": int(yr)}
        compounded = 1.0
        for m_idx, m_name in enumerate(_MONTH_NAMES, start=1):
            if m_idx in month_map:
                m_ret = month_map[m_idx]
                r_dict[m_name] = float(m_ret)
                compounded *= 1.0 + (m_ret / 100.0)
            else:
                r_dict[m_name] = None
        ytd_pct = (compounded - 1.0) * 100.0
        if abs(ytd_pct) <= 1e-12:
            ytd_pct = 0.0
        r_dict["YTD"] = float(ytd_pct)
        matrix_rows.append(r_dict)

    return pd.DataFrame(matrix_rows, columns=MONTHLY_MATRIX_COLUMNS)


def analyze_custom_regimes(
    equity_df: pd.DataFrame | None,
    trades_df: pd.DataFrame | None,
    regimes: list[tuple[str, Any, Any] | dict[str, Any]] | None,
    ledger_df: pd.DataFrame | None = None,
    initial_balance: float | None = None,
) -> pd.DataFrame:
    """
    Phân tích hiệu suất theo các giai đoạn thị trường tùy chỉnh (Custom Regimes).
    Mỗi phần tử trong `regimes` là tuple `(regime_name, start_time, end_time)`
    hoặc dict `{"name": ..., "start_time": ..., "end_time": ...}`.
    """
    if equity_df is None or equity_df.empty or not regimes:
        return pd.DataFrame(columns=REGIME_STATS_COLUMNS)

    eff_init = (
        float(initial_balance)
        if initial_balance is not None
        else float(equity_df["equity"].iloc[0])
    )
    ret_df = compute_return_series(
        equity_df, ledger_df=ledger_df, initial_balance=eff_init
    )
    ret_df["time_dt"] = pd.to_datetime(ret_df["time"])

    closed_trades = pd.DataFrame()
    if trades_df is not None and not trades_df.empty and "exit_time" in trades_df.columns:
        closed_trades = trades_df[trades_df["exit_time"].notna()].copy()
        if not closed_trades.empty:
            closed_trades["_exit_dt"] = pd.to_datetime(closed_trades["exit_time"])

    rows: list[dict[str, Any]] = []
    for item in regimes:
        if isinstance(item, dict):
            r_name = str(item.get("name") or item.get("regime") or "REGIME")
            s_raw = item["start_time"]
            e_raw = item["end_time"]
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            r_name, s_raw, e_raw = str(item[0]), item[1], item[2]
        else:
            raise ValueError(
                "Mỗi phần tử trong regimes phải là (name, start_time, end_time) hoặc dict tương ứng."
            )

        s_ts = pd.Timestamp(s_raw)
        e_ts = pd.Timestamp(e_raw)
        if s_ts > e_ts:
            raise ValueError(f"Regime '{r_name}' có start_time ({s_ts}) > end_time ({e_ts}).")

        mask = (ret_df["time_dt"] >= s_ts) & (ret_df["time_dt"] <= e_ts)
        sub = ret_df[mask]
        if sub.empty:
            rows.append(
                {
                    "regime": r_name,
                    "start_time": s_ts,
                    "end_time": e_ts,
                    "start_equity": float("nan"),
                    "end_equity": float("nan"),
                    "return_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate_pct": 0.0,
                    "net_pnl": 0.0,
                    "profit_factor": 0.0,
                }
            )
            continue

        first_idx = int(sub.index[0])
        if first_idx > 0:
            anchor_wealth = float(ret_df.loc[first_idx - 1, "wealth_index"])
            start_eq = float(ret_df.loc[first_idx - 1, "equity"])
            twr_slice = sub["twr_return"].to_numpy(dtype=float)
            growth = float(np.prod(1.0 + twr_slice))
        else:
            anchor_wealth = float(sub["wealth_index"].iloc[0])
            start_eq = float(sub["equity"].iloc[0])
            w_end = float(sub["wealth_index"].iloc[-1])
            growth = (w_end / anchor_wealth) if anchor_wealth > 0.0 else 1.0

        end_eq = float(sub["equity"].iloc[-1])
        ret_pct = (growth - 1.0) * 100.0
        if abs(ret_pct) <= 1e-12:
            ret_pct = 0.0

        wealth_arr = sub["wealth_index"].to_numpy(dtype=float)
        local_dd = _compute_local_drawdown_pct(wealth_arr, anchor_wealth)

        if not closed_trades.empty:
            t_mask = (closed_trades["_exit_dt"] >= s_ts) & (closed_trades["_exit_dt"] <= e_ts)
            r_trades = closed_trades[t_mask]
        else:
            r_trades = pd.DataFrame()

        t_stats = analyze_trades(r_trades)
        rows.append(
            {
                "regime": r_name,
                "start_time": sub["time"].iloc[0],
                "end_time": sub["time"].iloc[-1],
                "start_equity": start_eq,
                "end_equity": end_eq,
                "return_pct": float(ret_pct),
                "max_drawdown_pct": float(local_dd),
                "trades": int(t_stats["total_trades"]),
                "wins": int(t_stats["wins"]),
                "losses": int(t_stats["losses"]),
                "win_rate_pct": float(t_stats["win_rate_pct"]),
                "net_pnl": float(t_stats["total_net_pnl"]),
                "profit_factor": float(t_stats["profit_factor"]),
            }
        )

    return pd.DataFrame(rows, columns=REGIME_STATS_COLUMNS)
