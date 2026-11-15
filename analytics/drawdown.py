from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

from quant_backtester.analytics.returns import compute_return_series


DRAWDOWN_SERIES_COLUMNS = [
    "time",
    "equity",
    "wealth_index",
    "peak_wealth",
    "drawdown_pct",
    "drawdown_depth_pct",
    "drawdown_amount",
    "drawdown_depth",
    "account_peak_equity",
    "account_drawdown_amount",
    "account_drawdown_depth",
    "account_drawdown_pct",
    "underwater_bars",
]

DRAWDOWN_EPISODE_COLUMNS = [
    "episode_id",
    "peak_time",
    "peak_bar_index",
    "peak_value",
    "valley_time",
    "valley_bar_index",
    "valley_value",
    "recovery_time",
    "recovery_bar_index",
    "max_drawdown_pct",
    "max_drawdown_amount",
    "decline_bars",
    "recovery_bars",
    "duration_bars",
    "duration_time",
    "is_recovered",
]


def compute_drawdown_series(
    equity_df: pd.DataFrame | None,
    ledger_df: pd.DataFrame | None = None,
    initial_balance: float | None = None,
) -> pd.DataFrame:
    """
    Tính bảng Drawdown theo từng Bar, tách biệt rõ:
    1. Performance Drawdown (trên Wealth Index đã loại bỏ nhiễu Deposit/Withdrawal).
    2. Account Equity Drawdown (trên Equity thực tế có điều chỉnh đỉnh HWM theo dòng tiền).
    """
    if equity_df is None or equity_df.empty:
        return pd.DataFrame(columns=DRAWDOWN_SERIES_COLUMNS)

    ret_df = compute_return_series(
        equity_df, ledger_df=ledger_df, initial_balance=initial_balance
    )
    n = len(ret_df)
    equities = ret_df["equity"].to_numpy(dtype=float)
    wealth = ret_df["wealth_index"].to_numpy(dtype=float)
    cfs = ret_df["external_cash_flow"].to_numpy(dtype=float)
    eff_init_bal = float(initial_balance) if initial_balance is not None else float(equities[0])

    peak_wealth = np.zeros(n, dtype=float)
    dd_pct = np.zeros(n, dtype=float)
    dd_depth_pct = np.zeros(n, dtype=float)
    dd_amount = np.zeros(n, dtype=float)
    dd_depth = np.zeros(n, dtype=float)

    acc_peak = np.zeros(n, dtype=float)
    acc_dd_amount = np.zeros(n, dtype=float)
    acc_dd_depth = np.zeros(n, dtype=float)
    acc_dd_pct = np.zeros(n, dtype=float)
    underwater = np.zeros(n, dtype=int)

    curr_peak_w = max(1.0, wealth[0]) if initial_balance is not None else wealth[0]
    curr_acc_hwm = max(eff_init_bal + cfs[0], equities[0])
    uw_count = 0

    for i in range(n):
        w_i = wealth[i]
        eq_i = equities[i]
        if i == 0:
            if w_i >= curr_peak_w - 1e-12:
                curr_peak_w = w_i
                uw_count = 0
            else:
                uw_count = 1
        else:
            curr_acc_hwm = max(curr_acc_hwm + cfs[i], eq_i)
            if w_i >= curr_peak_w - 1e-12:
                curr_peak_w = max(curr_peak_w, w_i)
                uw_count = 0
            else:
                uw_count += 1

        peak_wealth[i] = curr_peak_w
        if curr_peak_w > 0.0:
            pct_val = ((w_i / curr_peak_w) - 1.0) * 100.0
            if abs(pct_val) <= 1e-12:
                pct_val = 0.0
        else:
            pct_val = 0.0
        dd_pct[i] = min(0.0, pct_val)
        dd_depth_pct[i] = abs(dd_pct[i])

        scaled_amt = eff_init_bal * (w_i - curr_peak_w)
        if abs(scaled_amt) <= 1e-12:
            scaled_amt = 0.0
        dd_amount[i] = min(0.0, scaled_amt)
        dd_depth[i] = abs(dd_amount[i])

        acc_peak[i] = curr_acc_hwm
        raw_acc_dd = eq_i - curr_acc_hwm
        if abs(raw_acc_dd) <= 1e-12:
            raw_acc_dd = 0.0
        acc_dd_amount[i] = min(0.0, raw_acc_dd)
        acc_dd_depth[i] = abs(acc_dd_amount[i])
        if curr_acc_hwm > 0.0:
            acc_pct_val = ((eq_i / curr_acc_hwm) - 1.0) * 100.0
            if abs(acc_pct_val) <= 1e-12:
                acc_pct_val = 0.0
        else:
            acc_pct_val = 0.0
        acc_dd_pct[i] = min(0.0, acc_pct_val)
        underwater[i] = uw_count

    return pd.DataFrame(
        {
            "time": ret_df["time"],
            "equity": equities,
            "wealth_index": wealth,
            "peak_wealth": peak_wealth,
            "drawdown_pct": dd_pct,
            "drawdown_depth_pct": dd_depth_pct,
            "drawdown_amount": dd_amount,
            "drawdown_depth": dd_depth,
            "account_peak_equity": acc_peak,
            "account_drawdown_amount": acc_dd_amount,
            "account_drawdown_depth": acc_dd_depth,
            "account_drawdown_pct": acc_dd_pct,
            "underwater_bars": underwater,
        },
        columns=DRAWDOWN_SERIES_COLUMNS,
    )


def extract_drawdown_episodes(drawdown_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Trích xuất danh sách các chu kỳ sụt giảm không chồng lấn (Peak -> Valley -> Recovery).
    Quy tắc tie-breaking xác định:
    - Nếu có nhiều Bar bằng nhau ở đỉnh trước khi giảm, lấy Bar đỉnh cuối cùng ngay trước khi giảm.
    - Nếu có nhiều Bar bằng nhau ở đáy trong cùng một Episode, lấy Bar đầu tiên chạm đáy sâu nhất đó.
    """
    if drawdown_df is None or drawdown_df.empty:
        return pd.DataFrame(columns=DRAWDOWN_EPISODE_COLUMNS)

    times = pd.to_datetime(drawdown_df["time"]).tolist()
    equities = drawdown_df["equity"].astype(float).to_numpy()
    dd_pcts = drawdown_df["drawdown_pct"].astype(float).to_numpy()
    acc_depths = (
        drawdown_df["account_drawdown_depth"].astype(float).to_numpy()
        if "account_drawdown_depth" in drawdown_df.columns
        else drawdown_df["drawdown_depth"].astype(float).to_numpy()
    )
    underwater = drawdown_df["underwater_bars"].astype(int).to_numpy()

    episodes: list[dict[str, Any]] = []
    in_episode = False
    last_peak_idx = 0

    ep_peak_idx = 0
    ep_valley_idx = 0
    ep_min_dd_pct = 0.0
    ep_max_dd_amt = 0.0

    n = len(drawdown_df)
    for i in range(n):
        uw = int(underwater[i])
        if uw == 0:
            if in_episode:
                # Episode kết thúc và hồi phục tại Bar i
                ep_id = len(episodes) + 1
                decline_b = ep_valley_idx - ep_peak_idx
                recovery_b = i - ep_valley_idx
                dur_b = (i - 1) - ep_peak_idx
                dur_time = times[i] - times[ep_peak_idx]
                episodes.append(
                    {
                        "episode_id": ep_id,
                        "peak_time": times[ep_peak_idx],
                        "peak_bar_index": int(ep_peak_idx),
                        "peak_value": float(equities[ep_peak_idx]),
                        "valley_time": times[ep_valley_idx],
                        "valley_bar_index": int(ep_valley_idx),
                        "valley_value": float(equities[ep_valley_idx]),
                        "recovery_time": times[i],
                        "recovery_bar_index": int(i),
                        "max_drawdown_pct": float(ep_min_dd_pct),
                        "max_drawdown_amount": float(ep_max_dd_amt),
                        "decline_bars": int(decline_b),
                        "recovery_bars": int(recovery_b),
                        "duration_bars": int(dur_b),
                        "duration_time": dur_time,
                        "is_recovered": True,
                    }
                )
                in_episode = False
            last_peak_idx = i
        else:
            if not in_episode:
                in_episode = True
                ep_peak_idx = last_peak_idx
                ep_valley_idx = i
                ep_min_dd_pct = float(dd_pcts[i])
                ep_max_dd_amt = float(acc_depths[i])
            else:
                if float(dd_pcts[i]) < ep_min_dd_pct - 1e-12:
                    ep_min_dd_pct = float(dd_pcts[i])
                    ep_valley_idx = i
                if float(acc_depths[i]) > ep_max_dd_amt + 1e-12:
                    ep_max_dd_amt = float(acc_depths[i])

    if in_episode:
        ep_id = len(episodes) + 1
        last_idx = n - 1
        decline_b = ep_valley_idx - ep_peak_idx
        dur_b = int(underwater[last_idx])
        dur_time = times[last_idx] - times[ep_peak_idx]
        episodes.append(
            {
                "episode_id": ep_id,
                "peak_time": times[ep_peak_idx],
                "peak_bar_index": int(ep_peak_idx),
                "peak_value": float(equities[ep_peak_idx]),
                "valley_time": times[ep_valley_idx],
                "valley_bar_index": int(ep_valley_idx),
                "valley_value": float(equities[ep_valley_idx]),
                "recovery_time": None,
                "recovery_bar_index": None,
                "max_drawdown_pct": float(ep_min_dd_pct),
                "max_drawdown_amount": float(ep_max_dd_amt),
                "decline_bars": int(decline_b),
                "recovery_bars": None,
                "duration_bars": int(dur_b),
                "duration_time": dur_time,
                "is_recovered": False,
            }
        )

    if not episodes:
        return pd.DataFrame(columns=DRAWDOWN_EPISODE_COLUMNS)
    return pd.DataFrame(episodes, columns=DRAWDOWN_EPISODE_COLUMNS)


def summarize_drawdown(
    drawdown_df: pd.DataFrame | None,
    net_trading_profit: float = 0.0,
    episodes_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Tổng hợp các chỉ số Drawdown:
    - max_drawdown_pct (<= 0.0, giữ quy ước âm tương thích ngược)
    - max_drawdown_depth_pct (>= 0.0)
    - max_drawdown_amount (>= 0.0, theo đơn vị tiền tệ tài khoản đã điều chỉnh dòng tiền)
    - avg_drawdown_pct (<= 0.0, trung bình trên toàn bộ các Bar)
    - avg_episode_drawdown_pct (<= 0.0, trung bình mức sụt giảm lớn nhất của từng Episode)
    - max_drawdown_duration_bars (số Bar liên tục dưới Peak dài nhất)
    - max_drawdown_duration_time (pd.Timedelta dài nhất của các Episode)
    - ulcer_index (căn bậc hai trung bình bình phương drawdown_pct)
    - recovery_factor (net_trading_profit / max_drawdown_amount)
    """
    if drawdown_df is None or drawdown_df.empty:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_depth_pct": 0.0,
            "max_drawdown_amount": 0.0,
            "avg_drawdown_pct": 0.0,
            "avg_episode_drawdown_pct": 0.0,
            "max_drawdown_duration_bars": 0,
            "max_drawdown_duration_time": pd.Timedelta(0),
            "ulcer_index": 0.0,
            "recovery_factor": 0.0,
            "drawdown_episodes_count": 0,
        }

    if episodes_df is None:
        episodes_df = extract_drawdown_episodes(drawdown_df)

    dd_pct = drawdown_df["drawdown_pct"].astype(float).to_numpy()
    acc_depth = (
        drawdown_df["account_drawdown_depth"].astype(float).to_numpy()
        if "account_drawdown_depth" in drawdown_df.columns
        else drawdown_df["drawdown_depth"].astype(float).to_numpy()
    )
    uw_bars = drawdown_df["underwater_bars"].astype(int).to_numpy()

    min_dd_pct = float(np.min(dd_pct)) if len(dd_pct) > 0 else 0.0
    if abs(min_dd_pct) <= 1e-12:
        min_dd_pct = 0.0
    max_dd_depth_pct = abs(min_dd_pct)
    max_dd_amount = float(np.max(acc_depth)) if len(acc_depth) > 0 else 0.0
    avg_dd_pct = float(np.mean(dd_pct)) if len(dd_pct) > 0 else 0.0
    if abs(avg_dd_pct) <= 1e-12:
        avg_dd_pct = 0.0

    ulcer = float(math.sqrt(float(np.mean(np.square(dd_pct))))) if len(dd_pct) > 0 else 0.0
    max_dur_bars = int(np.max(uw_bars)) if len(uw_bars) > 0 else 0

    if episodes_df is not None and not episodes_df.empty:
        avg_ep_dd = float(episodes_df["max_drawdown_pct"].astype(float).mean())
        max_dur_time = episodes_df["duration_time"].max()
        ep_count = int(len(episodes_df))
    else:
        avg_ep_dd = 0.0
        max_dur_time = pd.Timedelta(0)
        ep_count = 0

    if max_dd_amount > 1e-12:
        rec_factor = float(net_trading_profit) / max_dd_amount
    else:
        rec_factor = float("inf") if float(net_trading_profit) > 1e-12 else 0.0

    return {
        "max_drawdown_pct": min_dd_pct,
        "max_drawdown_depth_pct": max_dd_depth_pct,
        "max_drawdown_amount": max_dd_amount,
        "avg_drawdown_pct": avg_dd_pct,
        "avg_episode_drawdown_pct": avg_ep_dd,
        "max_drawdown_duration_bars": max_dur_bars,
        "max_drawdown_duration_time": max_dur_time,
        "ulcer_index": ulcer,
        "recovery_factor": rec_factor,
        "drawdown_episodes_count": ep_count,
    }
