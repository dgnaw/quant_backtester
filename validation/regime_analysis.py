"""H8 – Automated Market Regime Classification & Performance Breakdown."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd

from quant_backtester.analytics.drawdown import compute_drawdown_series
from quant_backtester.analytics.returns import compute_return_series
from quant_backtester.analytics.trade_analysis import analyze_trades
from quant_backtester.validation.config import MarketRegimeConfig
from quant_backtester.validation.models import ValidationStatus

REGIME_STATS_COLUMNS = [
    "regime",
    "bars_count",
    "time_share_pct",
    "regime_return_pct",
    "max_drawdown_pct",
    "trades",
    "wins",
    "losses",
    "win_rate_pct",
    "net_pnl",
    "profit_factor",
    "trade_attribution_method",
]


def _short_trend_label(trend_label: str) -> str:
    if trend_label == "BULL_TREND":
        return "BULL"
    if trend_label == "BEAR_TREND":
        return "BEAR"
    if trend_label == "SIDEWAYS_RANGE":
        return "SIDEWAYS"
    return "UNKNOWN"


def classify_market_regimes(
    market_data: pd.DataFrame,
    regime_config: MarketRegimeConfig | None = None,
) -> pd.DataFrame:
    """
    Phân loại điều kiện thị trường thuần nhân quả tại từng nến t (chỉ dùng dữ liệu <= t):
    - TrendScore_t = sum(r_{t-L+1..t}) / (std(r_{t-L+1..t}) * sqrt(L))
    - RV_t = std(r_{t-V+1..t}) so sánh với ngưỡng nhân quả (Expanding / Rolling / Fixed).
    """
    if market_data is None or market_data.empty:
        return pd.DataFrame(
            columns=["time", "close", "log_return", "trend_score", "rolling_vol", "vol_threshold", "trend_regime", "vol_regime", "combined_regime"]
        )
    if "time" not in market_data.columns or "close" not in market_data.columns:
        raise ValueError("market_data phải có cột 'time' và 'close'.")

    cfg = regime_config if regime_config is not None else MarketRegimeConfig()
    df = market_data[["time", "close"]].copy().reset_index(drop=True)
    closes = df["close"].astype(float)
    if (closes <= 0.0).any() or not np.all(np.isfinite(closes.to_numpy())):
        raise ValueError("Cột 'close' phải chứa giá trị dương hữu hạn.")

    log_ret = np.log(closes / closes.shift(1))
    L = int(cfg.trend_window)
    V = int(cfg.vol_window)

    roll_sum = log_ret.rolling(window=L, min_periods=L).sum()
    roll_std_trend = log_ret.rolling(window=L, min_periods=L).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        trend_score = pd.Series(
            np.where(
                roll_std_trend.notna() & (roll_std_trend > 1e-12),
                roll_sum / (roll_std_trend * math.sqrt(float(L))),
                np.nan,
            ),
            index=df.index,
            dtype=float,
        )

    rolling_vol = log_ret.rolling(window=V, min_periods=V).std(ddof=1)

    min_hist = cfg.min_vol_history if cfg.min_vol_history is not None else 1
    if cfg.vol_threshold_method == "FIXED":
        vol_thresh = pd.Series(float(cfg.fixed_vol_threshold), index=df.index, dtype=float)
    elif cfg.vol_threshold_method == "EXPANDING_MEDIAN":
        vol_thresh = rolling_vol.expanding(min_periods=min_hist).median()
    elif cfg.vol_threshold_method == "ROLLING_MEDIAN":
        vol_thresh = rolling_vol.rolling(window=cfg.vol_threshold_window, min_periods=min_hist).median()
    else:  # ROLLING_PERCENTILE
        vol_thresh = rolling_vol.rolling(window=cfg.vol_threshold_window, min_periods=min_hist).quantile(cfg.vol_percentile)

    trend_labels: list[str] = []
    vol_labels: list[str] = []
    combined_labels: list[str] = []

    for i in range(len(df)):
        ts = trend_score.iloc[i]
        if pd.isna(ts) or not math.isfinite(float(ts)):
            t_lbl = "UNKNOWN"
        elif float(ts) > cfg.trend_threshold:
            t_lbl = "BULL_TREND"
        elif float(ts) < -cfg.trend_threshold:
            t_lbl = "BEAR_TREND"
        else:
            t_lbl = "SIDEWAYS_RANGE"

        rv = rolling_vol.iloc[i]
        vt = vol_thresh.iloc[i]
        if pd.isna(rv) or pd.isna(vt) or not (math.isfinite(float(rv)) and math.isfinite(float(vt))):
            v_lbl = "UNKNOWN"
        elif float(rv) > float(vt):
            v_lbl = "HIGH_VOL"
        else:
            v_lbl = "LOW_VOL"

        if t_lbl != "UNKNOWN" and v_lbl != "UNKNOWN":
            c_lbl = f"{_short_trend_label(t_lbl)}_{v_lbl}"
        else:
            c_lbl = "UNKNOWN"

        trend_labels.append(t_lbl)
        vol_labels.append(v_lbl)
        combined_labels.append(c_lbl)

    return pd.DataFrame(
        {
            "time": df["time"],
            "close": closes,
            "log_return": log_ret,
            "trend_score": trend_score,
            "rolling_vol": rolling_vol,
            "vol_threshold": vol_thresh,
            "trend_regime": trend_labels,
            "vol_regime": vol_labels,
            "combined_regime": combined_labels,
        }
    )


def extract_contiguous_segments(
    labeled_df: pd.DataFrame,
    regime_col: str = "combined_regime",
) -> pd.DataFrame:
    """Gom các nến cùng nhãn Regime liên tiếp thành các đoạn (regime_label, start_time, end_time, bars_count)."""
    if labeled_df is None or labeled_df.empty:
        return pd.DataFrame(columns=["regime_label", "start_time", "end_time", "start_index", "end_index", "bars_count"])
    labels = labeled_df[regime_col].astype(str).tolist()
    times = labeled_df["time"].tolist()
    segments: list[dict[str, Any]] = []
    cur_lbl = labels[0]
    start_idx = 0
    for idx in range(1, len(labels)):
        if labels[idx] != cur_lbl:
            segments.append(
                {
                    "regime_label": cur_lbl,
                    "start_time": times[start_idx],
                    "end_time": times[idx - 1],
                    "start_index": start_idx,
                    "end_index": idx - 1,
                    "bars_count": idx - start_idx,
                }
            )
            cur_lbl = labels[idx]
            start_idx = idx
    segments.append(
        {
            "regime_label": cur_lbl,
            "start_time": times[start_idx],
            "end_time": times[-1],
            "start_index": start_idx,
            "end_index": len(labels) - 1,
            "bars_count": len(labels) - start_idx,
        }
    )
    return pd.DataFrame(segments)


def _aggregate_regime_stats(
    merged: pd.DataFrame,
    trades_df: pd.DataFrame,
    regime_col: str,
) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=REGIME_STATS_COLUMNS)

    total_bars = len(merged)
    has_trades = isinstance(trades_df, pd.DataFrame) and not trades_df.empty and "exit_time" in trades_df.columns
    if has_trades:
        t_work = trades_df[trades_df["exit_time"].notna()].copy()
        t_work["_exit_dt"] = pd.to_datetime(t_work["exit_time"])
        # Gán Regime cho mỗi Trade theo quy tắc Exit Time khớp với mốc nến gần nhất <= exit_time
        bar_map = merged[["_time_dt", regime_col]].sort_values("_time_dt")
        t_work = pd.merge_asof(
            t_work.sort_values("_exit_dt"),
            bar_map.rename(columns={"_time_dt": "_exit_dt", regime_col: "_trade_regime"}),
            on="_exit_dt",
            direction="backward",
        )
    else:
        t_work = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for label, grp in merged.groupby(regime_col, sort=True):
        raw_twr = grp["twr_return"].astype(float).to_numpy()
        twr_arr = raw_twr[np.isfinite(raw_twr)]
        if len(twr_arr) > 0:
            growth = float(np.prod(1.0 + twr_arr))
            ret_pct = (growth - 1.0) * 100.0
            if abs(ret_pct) <= 1e-12:
                ret_pct = 0.0
            eq_vals = np.maximum(0.0, np.concatenate([[1.0], np.cumprod(1.0 + twr_arr)]))
            peaks = np.maximum.accumulate(eq_vals)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd_vals = np.where(peaks > 0.0, (eq_vals / peaks - 1.0) * 100.0, 0.0)
            max_dd = float(np.min(dd_vals))
        else:
            ret_pct = 0.0
            max_dd = 0.0

        r_trades = t_work[t_work["_trade_regime"] == label] if not t_work.empty else pd.DataFrame()
        t_stats = analyze_trades(r_trades)
        rows.append(
            {
                "regime": str(label),
                "bars_count": int(len(grp)),
                "time_share_pct": float(len(grp) / total_bars * 100.0),
                "regime_return_pct": float(ret_pct),
                "max_drawdown_pct": float(max_dd),
                "trades": int(t_stats["total_trades"]),
                "wins": int(t_stats["wins"]),
                "losses": int(t_stats["losses"]),
                "win_rate_pct": float(t_stats["win_rate_pct"]),
                "net_pnl": float(t_stats["total_net_pnl"]),
                "profit_factor": float(t_stats["profit_factor"]),
                "trade_attribution_method": "EXIT_TIME",
            }
        )
    return pd.DataFrame(rows, columns=REGIME_STATS_COLUMNS)


@dataclass
class MarketRegimeReport:
    status: ValidationStatus
    labeled_bars_df: pd.DataFrame
    segments_df: pd.DataFrame
    trend_regime_stats: pd.DataFrame
    vol_regime_stats: pd.DataFrame
    combined_regime_stats: pd.DataFrame
    trade_attribution_method: str = "EXIT_TIME"
    messages: list[str] = field(default_factory=list)


def analyze_market_regimes(
    market_data: pd.DataFrame,
    result: Any,
    regime_config: MarketRegimeConfig | None = None,
) -> MarketRegimeReport:
    """
    Phân loại thị trường theo Trend & Volatility rồi tổng hợp hiệu suất chiến lược theo từng Regime.
    Phân biệt rõ Regime Return (tính từ TWR chuỗi nến) và Trade Statistics (gán theo Exit Time).
    """
    labeled = classify_market_regimes(market_data, regime_config=regime_config)
    if labeled.empty:
        empty = pd.DataFrame(columns=REGIME_STATS_COLUMNS)
        return MarketRegimeReport(
            status=ValidationStatus.NOT_RUN,
            labeled_bars_df=labeled,
            segments_df=pd.DataFrame(),
            trend_regime_stats=empty,
            vol_regime_stats=empty,
            combined_regime_stats=empty,
            messages=["EMPTY_DATA: Không có dữ liệu để phân tích Regime."],
        )

    equity_df = result.equity if hasattr(result, "equity") else result["equity"]
    trades_df = result.trades if hasattr(result, "trades") else result.get("trades", pd.DataFrame())
    ledger_df = result.ledger if hasattr(result, "ledger") else result.get("ledger")
    cfg = getattr(result, "config", None)
    init_bal = float(cfg.account.initial_balance) if cfg is not None else float(equity_df["equity"].iloc[0])

    ret_df = compute_return_series(equity_df, ledger_df=ledger_df, initial_balance=init_bal)
    ret_df["_time_dt"] = pd.to_datetime(ret_df["time"])
    labeled["_time_dt"] = pd.to_datetime(labeled["time"])

    merged = pd.merge(
        labeled,
        ret_df[["_time_dt", "twr_return"]],
        on="_time_dt",
        how="inner",
    )
    segments = extract_contiguous_segments(labeled, regime_col="combined_regime")
    trend_stats = _aggregate_regime_stats(merged, trades_df, "trend_regime")
    vol_stats = _aggregate_regime_stats(merged, trades_df, "vol_regime")
    comb_stats = _aggregate_regime_stats(merged, trades_df, "combined_regime")

    known_bars = int((labeled["combined_regime"] != "UNKNOWN").sum())
    status = ValidationStatus.PASS if known_bars > 0 else ValidationStatus.WARNING
    msgs = [] if known_bars > 0 else ["INSUFFICIENT_WARMUP_FOR_REGIME: Toàn bộ nến có nhãn UNKNOWN."]
    return MarketRegimeReport(
        status=status,
        labeled_bars_df=labeled.drop(columns=["_time_dt"]),
        segments_df=segments,
        trend_regime_stats=trend_stats,
        vol_regime_stats=vol_stats,
        combined_regime_stats=comb_stats,
        messages=msgs,
    )
