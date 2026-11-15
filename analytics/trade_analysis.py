from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


def _compute_streaks(net_pnls: np.ndarray) -> tuple[int, int]:
    max_wins = 0
    max_losses = 0
    curr_wins = 0
    curr_losses = 0
    for val in net_pnls:
        if val > 0.0:
            curr_wins += 1
            curr_losses = 0
            if curr_wins > max_wins:
                max_wins = curr_wins
        elif val < 0.0:
            curr_losses += 1
            curr_wins = 0
            if curr_losses > max_losses:
                max_losses = curr_losses
        else:
            # Breakeven (net_pnl == 0) kết thúc chuỗi thắng/thua hiện tại
            curr_wins = 0
            curr_losses = 0
    return max_wins, max_losses


def analyze_trades(trades_df: pd.DataFrame | None) -> dict[str, Any]:
    """
    Phân tích toàn diện danh sách giao dịch đã đóng (Closed Trades) dựa trên Net PnL:
    - Phân loại Win / Loss / Breakeven trên net_pnl.
    - Tính Profit Factor, Expectancy ($), Payoff Ratio, Avg Win / Avg Loss, Largest Win / Loss.
    - Tính R-Multiple, Average R, Expectancy R, r_coverage_pct và SQN chuẩn (trên R-Multiple).
    - Cung cấp tách biệt empirical_r_multiple và sqn_cash_proxy.
    - Đối chiếu Gross PnL vs Net PnL, Total Fees, Total Slippage Cost (chẩn đoán, không trừ lại lần 2) và Cost Drag.
    - Phân tích chuỗi thắng/thua liên tiếp, Long/Short Breakdown và Holding Bars.
    """
    empty_result: dict[str, Any] = {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "breakevens": 0,
        "win_rate_pct": 0.0,
        "loss_rate_pct": 0.0,
        "breakeven_rate_pct": 0.0,
        "profit_factor": 0.0,
        "gross_profit_factor": 0.0,
        "expectancy_cash_per_trade": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "payoff_ratio": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "total_gross_pnl": 0.0,
        "total_net_pnl": 0.0,
        "total_fees": 0.0,
        "total_slippage_cost": 0.0,
        "cost_drag": 0.0,
        "cost_drag_pct": None,
        "valid_r_trades": 0,
        "r_coverage_pct": 0.0,
        "average_r": None,
        "expectancy_r": None,
        "r_win_avg": None,
        "r_loss_avg": None,
        "sqn": None,
        "empirical_r_multiple": None,
        "sqn_cash_proxy": None,
        "long_trades": 0,
        "short_trades": 0,
        "long_win_rate_pct": 0.0,
        "short_win_rate_pct": 0.0,
        "long_net_pnl": 0.0,
        "short_net_pnl": 0.0,
        "avg_holding_bars": None,
        "avg_winner_holding_bars": None,
        "avg_loser_holding_bars": None,
    }

    if trades_df is None or trades_df.empty:
        return empty_result

    df = trades_df.copy()
    # Chỉ tính các giao dịch đã đóng (có exit_time hợp lệ nếu cột exit_time tồn tại)
    if "exit_time" in df.columns:
        df = df[df["exit_time"].notna()].copy()
    if df.empty:
        return empty_result

    # Sắp xếp ổn định theo thứ tự đóng lệnh
    sort_cols: list[str] = []
    if "exit_time" in df.columns:
        df["_sort_exit_time"] = pd.to_datetime(df["exit_time"])
        sort_cols.append("_sort_exit_time")
    if "exit_bar_index" in df.columns and df["exit_bar_index"].notna().any():
        sort_cols.append("exit_bar_index")
    if "trade_id" in df.columns:
        sort_cols.append("trade_id")
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if "net_pnl" not in df.columns:
        raise ValueError("trades_df phải có cột 'net_pnl'.")

    net_pnl = df["net_pnl"].astype(float).to_numpy()
    total_trades = len(net_pnl)

    wins_mask = net_pnl > 0.0
    losses_mask = net_pnl < 0.0
    be_mask = net_pnl == 0.0

    wins = int(np.sum(wins_mask))
    losses = int(np.sum(losses_mask))
    breakevens = int(np.sum(be_mask))

    win_rate_pct = float(wins / total_trades * 100.0)
    loss_rate_pct = float(losses / total_trades * 100.0)
    be_rate_pct = float(breakevens / total_trades * 100.0)

    pos_net_sum = float(np.sum(net_pnl[wins_mask])) if wins > 0 else 0.0
    neg_net_sum = float(-np.sum(net_pnl[losses_mask])) if losses > 0 else 0.0

    if neg_net_sum > 0.0:
        profit_factor = float(pos_net_sum / neg_net_sum)
    else:
        profit_factor = float("inf") if pos_net_sum > 0.0 else 0.0

    expectancy_cash = float(np.mean(net_pnl))
    avg_win = float(np.mean(net_pnl[wins_mask])) if wins > 0 else 0.0
    avg_loss = float(np.mean(net_pnl[losses_mask])) if losses > 0 else 0.0

    if abs(avg_loss) > 0.0:
        payoff_ratio = float(avg_win / abs(avg_loss))
    else:
        payoff_ratio = float("inf") if avg_win > 0.0 else 0.0

    largest_win = float(np.max(net_pnl))
    largest_loss = float(np.min(net_pnl))
    max_consec_wins, max_consec_losses = _compute_streaks(net_pnl)

    # Gross PnL, Fees & Slippage (Diagnostic)
    total_net_pnl = float(np.sum(net_pnl))
    if "total_fee" in df.columns:
        total_fees = float(df["total_fee"].astype(float).sum())
    elif "entry_fee" in df.columns and "exit_fee" in df.columns:
        total_fees = float(
            (df["entry_fee"].astype(float) + df["exit_fee"].astype(float)).sum()
        )
    else:
        total_fees = 0.0

    if "gross_pnl" in df.columns:
        gross_pnl = df["gross_pnl"].astype(float).to_numpy()
        total_gross_pnl = float(np.sum(gross_pnl))
        g_pos = float(np.sum(gross_pnl[gross_pnl > 0.0]))
        g_neg = float(-np.sum(gross_pnl[gross_pnl < 0.0]))
        gross_pf = float(g_pos / g_neg) if g_neg > 0.0 else (float("inf") if g_pos > 0.0 else 0.0)
    else:
        total_gross_pnl = total_net_pnl + total_fees
        gross_pf = profit_factor

    entry_slip = (
        df["entry_slippage_cost"].astype(float).sum()
        if "entry_slippage_cost" in df.columns
        else 0.0
    )
    exit_slip = (
        df["exit_slippage_cost"].astype(float).sum()
        if "exit_slippage_cost" in df.columns
        else 0.0
    )
    total_slippage_cost = float(entry_slip + exit_slip)

    cost_drag = float(total_gross_pnl - total_net_pnl)
    cost_drag_pct = (
        float((cost_drag / abs(total_gross_pnl)) * 100.0)
        if abs(total_gross_pnl) > 1e-12
        else None
    )

    # R-Multiple, Average R, Expectancy R & SQN
    r_values: list[float] = []
    if "r_multiple" in df.columns or "initial_risk_amount" in df.columns:
        for idx, row in df.iterrows():
            r_val = row.get("r_multiple") if "r_multiple" in df.columns else None
            if r_val is not None and not pd.isna(r_val) and math.isfinite(float(r_val)):
                r_values.append(float(r_val))
            elif "initial_risk_amount" in df.columns:
                init_r = row.get("initial_risk_amount")
                if (
                    init_r is not None
                    and not pd.isna(init_r)
                    and math.isfinite(float(init_r))
                    and float(init_r) > 1e-12
                ):
                    r_values.append(float(row["net_pnl"]) / float(init_r))

    valid_r_arr = np.array(r_values, dtype=float)
    valid_r_trades = len(valid_r_arr)
    r_coverage_pct = float(valid_r_trades / total_trades * 100.0)

    if valid_r_trades > 0:
        average_r: float | None = float(np.mean(valid_r_arr))
        expectancy_r: float | None = average_r
        r_wins = valid_r_arr[valid_r_arr > 0.0]
        r_losses = valid_r_arr[valid_r_arr < 0.0]
        r_win_avg: float | None = float(np.mean(r_wins)) if len(r_wins) > 0 else 0.0
        r_loss_avg: float | None = float(np.mean(r_losses)) if len(r_losses) > 0 else 0.0
        if valid_r_trades >= 2:
            r_std = float(np.std(valid_r_arr, ddof=1))
            sqn: float | None = (
                float((average_r / r_std) * math.sqrt(float(valid_r_trades)))
                if r_std > 1e-12
                else None
            )
        else:
            sqn = None
    else:
        average_r = None
        expectancy_r = None
        r_win_avg = None
        r_loss_avg = None
        sqn = None

    empirical_r_multiple: float | None = (
        float(expectancy_cash / abs(avg_loss))
        if (losses > 0 and abs(avg_loss) > 1e-12)
        else None
    )

    sqn_cash_proxy: float | None = None
    if total_trades >= 2:
        pnl_std = float(np.std(net_pnl, ddof=1))
        if pnl_std > 1e-12:
            sqn_cash_proxy = float((expectancy_cash / pnl_std) * math.sqrt(float(total_trades)))

    # Long / Short Breakdown
    if "direction" in df.columns:
        dirs = df["direction"].astype(str).str.upper()
        long_mask = dirs == "LONG"
        short_mask = dirs == "SHORT"
        long_pnls = net_pnl[long_mask.to_numpy()]
        short_pnls = net_pnl[short_mask.to_numpy()]
        long_trades = int(len(long_pnls))
        short_trades = int(len(short_pnls))
        long_win_rate_pct = (
            float(np.sum(long_pnls > 0.0) / long_trades * 100.0) if long_trades > 0 else 0.0
        )
        short_win_rate_pct = (
            float(np.sum(short_pnls > 0.0) / short_trades * 100.0) if short_trades > 0 else 0.0
        )
        long_net_pnl = float(np.sum(long_pnls)) if long_trades > 0 else 0.0
        short_net_pnl = float(np.sum(short_pnls)) if short_trades > 0 else 0.0
    else:
        long_trades = 0
        short_trades = 0
        long_win_rate_pct = 0.0
        short_win_rate_pct = 0.0
        long_net_pnl = 0.0
        short_net_pnl = 0.0

    # Holding Bars Breakdown
    hb_series: pd.Series | None = None
    if "weighted_holding_bars" in df.columns and df["weighted_holding_bars"].notna().any():
        hb_series = pd.to_numeric(df["weighted_holding_bars"], errors="coerce")
    elif "holding_bars" in df.columns and df["holding_bars"].notna().any():
        hb_series = pd.to_numeric(df["holding_bars"], errors="coerce")

    if hb_series is not None and hb_series.notna().any():
        avg_hb: float | None = float(hb_series.dropna().mean())
        win_hb = hb_series[wins_mask].dropna()
        loss_hb = hb_series[losses_mask].dropna()
        avg_win_hb: float | None = float(win_hb.mean()) if not win_hb.empty else None
        avg_loss_hb: float | None = float(loss_hb.mean()) if not loss_hb.empty else None
    else:
        avg_hb = None
        avg_win_hb = None
        avg_loss_hb = None

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "win_rate_pct": win_rate_pct,
        "loss_rate_pct": loss_rate_pct,
        "breakeven_rate_pct": be_rate_pct,
        "profit_factor": profit_factor,
        "gross_profit_factor": gross_pf,
        "expectancy_cash_per_trade": expectancy_cash,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "total_gross_pnl": total_gross_pnl,
        "total_net_pnl": total_net_pnl,
        "total_fees": total_fees,
        "total_slippage_cost": total_slippage_cost,
        "cost_drag": cost_drag,
        "cost_drag_pct": cost_drag_pct,
        "valid_r_trades": valid_r_trades,
        "r_coverage_pct": r_coverage_pct,
        "average_r": average_r,
        "expectancy_r": expectancy_r,
        "r_win_avg": r_win_avg,
        "r_loss_avg": r_loss_avg,
        "sqn": sqn,
        "empirical_r_multiple": empirical_r_multiple,
        "sqn_cash_proxy": sqn_cash_proxy,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_win_rate_pct": long_win_rate_pct,
        "short_win_rate_pct": short_win_rate_pct,
        "long_net_pnl": long_net_pnl,
        "short_net_pnl": short_net_pnl,
        "avg_holding_bars": avg_hb,
        "avg_winner_holding_bars": avg_win_hb,
        "avg_loser_holding_bars": avg_loss_hb,
    }
