"""H3 – Chronological In-Sample / Out-of-Sample Split & Evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import pandas as pd

from quant_backtester.config import EngineConfig
from quant_backtester.models.result import BacktestResult
from quant_backtester.validation.grid_runner import (
    instantiate_fresh_strategy,
    run_backtest,
)
from quant_backtester.validation.models import ValidationStatus
from quant_backtester.validation.warmup_guard import (
    EvaluationWindow,
    compute_evaluation_metrics,
)


@dataclass
class ForceCloseSummary:
    closed_positions_count: int
    total_force_close_pnl: float
    total_force_close_fee: float
    close_times: list[pd.Timestamp] = field(default_factory=list)


@dataclass
class TrainTestSplit:
    in_sample_data: pd.DataFrame
    out_of_sample_data: pd.DataFrame  # Có thể chứa warmup_prefix ở đầu
    oos_raw_data: pd.DataFrame        # OOS thuần, không có warmup_prefix
    split_index: int
    embargo_bars: int
    warmup_bars: int
    oos_trade_start: int
    is_start_time: pd.Timestamp
    is_end_time: pd.Timestamp
    oos_start_time: pd.Timestamp
    oos_end_time: pd.Timestamp


@dataclass
class TrainTestEvaluationResult:
    status: ValidationStatus
    split: TrainTestSplit
    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult
    in_sample_metrics: dict[str, Any]
    out_of_sample_metrics: dict[str, Any]
    is_window: EvaluationWindow
    oos_window: EvaluationWindow
    is_force_close_summary: ForceCloseSummary
    degradation_summary: dict[str, float | None]


def _validate_time_series(market_data: pd.DataFrame) -> pd.DataFrame:
    if market_data is None or not isinstance(market_data, pd.DataFrame) or market_data.empty:
        raise ValueError("market_data không được rỗng.")
    if "time" not in market_data.columns:
        raise ValueError("market_data phải có cột 'time'.")
    df = market_data.copy().reset_index(drop=True)
    times = pd.to_datetime(df["time"], errors="coerce")
    if times.isna().any():
        raise ValueError("market_data chứa timestamp không hợp lệ (NaT).")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Timestamp của market_data phải tăng dần nghiêm ngặt và không trùng lặp.")
    return df


def split_in_sample_out_of_sample(
    market_data: pd.DataFrame,
    *,
    train_ratio: float | None = 0.7,
    split_time: Any = None,
    split_index: int | None = None,
    warmup_bars: int = 0,
    embargo_bars: int = 0,
) -> TrainTestSplit:
    """
    Phân tách IS/OOS thuần theo trình tự thời gian với Embargo và Warm-up Prefix cho OOS.
    """
    df = _validate_time_series(market_data)
    n = len(df)
    if n < 2:
        raise ValueError("Cần ít nhất 2 nến để tách IS/OOS.")
    if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int) or warmup_bars < 0:
        raise ValueError("warmup_bars phải là số nguyên không âm.")
    if isinstance(embargo_bars, bool) or not isinstance(embargo_bars, int) or embargo_bars < 0:
        raise ValueError("embargo_bars phải là số nguyên không âm.")

    times = pd.to_datetime(df["time"])
    modes_given = sum(x is not None for x in (split_time, split_index))
    if modes_given > 1:
        raise ValueError("Chỉ được truyền một trong các tham số: split_time hoặc split_index.")

    if split_index is not None:
        if isinstance(split_index, bool) or not isinstance(split_index, int):
            raise ValueError("split_index phải là số nguyên.")
        s_idx = int(split_index)
    elif split_time is not None:
        st = pd.Timestamp(split_time)
        if pd.isna(st):
            raise ValueError("split_time không hợp lệ.")
        if times.dt.tz is not None and st.tzinfo is None:
            st = st.tz_localize(times.dt.tz)
        s_idx = int((times < st).sum())
    else:
        ratio = 0.7 if train_ratio is None else float(train_ratio)
        if not math.isfinite(ratio) or not (0.0 < ratio < 1.0):
            raise ValueError(f"train_ratio phải nằm trong (0, 1), nhận {train_ratio!r}.")
        s_idx = int(math.floor(n * ratio))

    oos_start_idx = s_idx + embargo_bars
    if s_idx <= 0 or oos_start_idx >= n:
        raise ValueError(
            f"Không đủ dữ liệu để tách IS/OOS (n={n}, split_index={s_idx}, embargo={embargo_bars})."
        )

    is_df = df.iloc[:s_idx].copy().reset_index(drop=True)
    oos_raw_df = df.iloc[oos_start_idx:].copy().reset_index(drop=True)

    prefix_start = max(0, oos_start_idx - warmup_bars)
    oos_trade_start = oos_start_idx - prefix_start
    oos_input_df = df.iloc[prefix_start:].copy().reset_index(drop=True)

    return TrainTestSplit(
        in_sample_data=is_df,
        out_of_sample_data=oos_input_df,
        oos_raw_data=oos_raw_df,
        split_index=s_idx,
        embargo_bars=embargo_bars,
        warmup_bars=warmup_bars,
        oos_trade_start=oos_trade_start,
        is_start_time=pd.Timestamp(is_df["time"].iloc[0]),
        is_end_time=pd.Timestamp(is_df["time"].iloc[-1]),
        oos_start_time=pd.Timestamp(oos_raw_df["time"].iloc[0]),
        oos_end_time=pd.Timestamp(oos_raw_df["time"].iloc[-1]),
    )


def summarize_force_closes(result: BacktestResult) -> ForceCloseSummary:
    trades = result.trades
    if trades is None or trades.empty or "reason" not in trades.columns:
        return ForceCloseSummary(0, 0.0, 0.0, [])
    fc = trades[trades["reason"].astype(str).str.contains("END_OF_TEST|FORCE_CLOSE", case=False, na=False)]
    if fc.empty:
        return ForceCloseSummary(0, 0.0, 0.0, [])
    pnl = float(fc["net_pnl"].astype(float).sum()) if "net_pnl" in fc.columns else 0.0
    fee = float(fc["total_fee"].astype(float).sum()) if "total_fee" in fc.columns else 0.0
    times = [pd.Timestamp(t) for t in fc["exit_time"].tolist()] if "exit_time" in fc.columns else []
    return ForceCloseSummary(len(fc), pnl, fee, times)


def _safe_ratio(num: Any, den: Any) -> float | None:
    if num is None or den is None:
        return None
    f_num, f_den = float(num), float(den)
    if not (math.isfinite(f_num) and math.isfinite(f_den)) or abs(f_den) <= 1e-12:
        return None
    return f_num / f_den


def run_in_sample_out_of_sample(
    market_data: pd.DataFrame,
    strategy: Any,
    config: EngineConfig | None = None,
    *,
    train_ratio: float | None = 0.7,
    split_time: Any = None,
    split_index: int | None = None,
    embargo_bars: int = 0,
    include_warmup_prefix: bool = True,
) -> TrainTestEvaluationResult:
    """
    Chạy độc lập In-Sample (với force_close_at_end=True để không rò rỉ Position)
    và Out-of-Sample (tính metrics thuần trên cửa sổ oos_raw_data, loại bỏ Warm-up Prefix).
    """
    probe = instantiate_fresh_strategy(strategy)
    warmup = int(getattr(probe, "warmup_period", 0) or 0)
    effective_warmup = warmup if include_warmup_prefix else 0

    split = split_in_sample_out_of_sample(
        market_data,
        train_ratio=train_ratio,
        split_time=split_time,
        split_index=split_index,
        warmup_bars=effective_warmup,
        embargo_bars=embargo_bars,
    )
    if len(split.in_sample_data) < warmup:
        raise ValueError(
            f"Tập IS ({len(split.in_sample_data)} nến) ngắn hơn warmup_period ({warmup})."
        )
    if len(split.out_of_sample_data) < warmup:
        raise ValueError(
            f"Tập OOS ({len(split.out_of_sample_data)} nến) ngắn hơn warmup_period ({warmup})."
        )

    # IS: bắt buộc force_close_at_end=True để không rò rỉ vị thế sang OOS
    is_result = run_backtest(split.in_sample_data, strategy, config=config, force_close_at_end=True)
    is_metrics, is_win = compute_evaluation_metrics(is_result)
    fc_summary = summarize_force_closes(is_result)

    # OOS: chạy với trade_start = oos_trade_start (khớp đúng đầu oos_raw_data)
    oos_result = run_backtest(
        split.out_of_sample_data,
        strategy,
        config=config,
        trade_start=split.oos_trade_start,
        force_close_at_end=True,
    )
    oos_metrics, oos_win = compute_evaluation_metrics(oos_result)

    degradation = {
        "return_degradation_ratio": _safe_ratio(oos_metrics.get("cagr_pct"), is_metrics.get("cagr_pct")),
        "sharpe_degradation_ratio": _safe_ratio(oos_metrics.get("sharpe_annual"), is_metrics.get("sharpe_annual")),
        "total_return_diff_pct": (
            float(oos_metrics["total_return_pct"]) - float(is_metrics["total_return_pct"])
            if oos_metrics.get("total_return_pct") is not None and is_metrics.get("total_return_pct") is not None
            else None
        ),
    }

    status = ValidationStatus.PASS
    if int(oos_metrics.get("total_trades", 0) or 0) == 0:
        status = ValidationStatus.WARNING

    return TrainTestEvaluationResult(
        status=status,
        split=split,
        in_sample_result=is_result,
        out_of_sample_result=oos_result,
        in_sample_metrics=is_metrics,
        out_of_sample_metrics=oos_metrics,
        is_window=is_win,
        oos_window=oos_win,
        is_force_close_summary=fc_summary,
        degradation_summary=degradation,
    )
