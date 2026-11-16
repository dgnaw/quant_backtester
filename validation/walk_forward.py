"""H4 – Strategy-Agnostic Walk-Forward Analysis & Chronological OOS Stitching."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_backtester.analytics.annualization import resolve_bars_per_year
from quant_backtester.analytics.drawdown import compute_drawdown_series
from quant_backtester.analytics.performance import calculate_cagr_pct
from quant_backtester.analytics.report import calculate_metrics
from quant_backtester.analytics.returns import compute_return_series
from quant_backtester.config import EngineConfig
from quant_backtester.strategies.base import StrategyMode
from quant_backtester.validation.grid_runner import (
    SUPPORTED_OBJECTIVES,
    clone_config,
    run_backtest,
    run_parameter_grid,
    select_best_run,
)
from quant_backtester.validation.models import ValidationStatus
from quant_backtester.validation.train_test import (
    ForceCloseSummary,
    _validate_time_series,
    summarize_force_closes,
)
from quant_backtester.validation.warmup_guard import compute_evaluation_metrics


class WalkForwardMode(Enum):
    ROLLING = "ROLLING"
    ANCHORED = "ANCHORED"


@dataclass
class WalkForwardFoldResult:
    fold_id: int
    mode: str
    train_start_index: int
    train_end_index: int
    test_start_index: int
    test_end_index: int
    embargo_bars: int
    oos_gap_bars: int
    train_start_time: pd.Timestamp
    train_end_time: pd.Timestamp
    train_bars: int
    test_start_time: pd.Timestamp
    test_end_time: pd.Timestamp
    test_bars: int
    best_params: dict[str, Any]
    train_objective_score: float
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    oos_force_close_summary: ForceCloseSummary
    oos_returns_slice: pd.DataFrame
    oos_trades: pd.DataFrame
    train_growth_factor: float | None = None
    train_return_bars: int = 0


@dataclass
class WalkForwardResult:
    status: ValidationStatus
    mode: WalkForwardMode
    objective_metric: str
    folds: list[WalkForwardFoldResult] = field(default_factory=list)
    folds_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stitched_oos_equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    stitched_oos_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    stitched_oos_metrics: dict[str, Any] = field(default_factory=dict)
    is_annualized_return_pct: float | None = None
    oos_annualized_return_pct: float | None = None
    walk_forward_efficiency: float | None = None
    messages: list[str] = field(default_factory=list)


def _generate_fold_windows(
    n_bars: int,
    *,
    folds: int,
    mode: WalkForwardMode,
    train_bars: int | None,
    test_bars: int | None,
    embargo_bars: int,
) -> list[tuple[int, int, int, int]]:
    """Trả về danh sách (train_start, train_end, test_start, test_end) không chồng lấp OOS."""
    windows: list[tuple[int, int, int, int]] = []
    if train_bars is not None or test_bars is not None:
        if train_bars is None or test_bars is None:
            raise ValueError("Phải truyền cả hai tham số train_bars và test_bars, hoặc để cả hai là None.")
        if train_bars <= 0 or test_bars <= 0:
            raise ValueError("train_bars và test_bars phải > 0.")
        pos = 0
        while True:
            tr_start = 0 if mode == WalkForwardMode.ANCHORED else pos
            tr_end = pos + train_bars
            te_start = tr_end + embargo_bars
            te_end = te_start + test_bars
            if te_end > n_bars:
                break
            windows.append((tr_start, tr_end, te_start, te_end))
            pos += test_bars
        return windows

    if folds < 2:
        raise ValueError("folds phải >= 2.")
    edges = np.linspace(0, n_bars, folds + 1, dtype=int)
    for k in range(folds - 1):
        tr_start = 0 if mode == WalkForwardMode.ANCHORED else int(edges[k])
        tr_end = int(edges[k + 1])
        te_start = tr_end + embargo_bars
        te_end = int(edges[k + 2])
        if tr_end > tr_start and te_end > te_start:
            windows.append((tr_start, tr_end, te_start, te_end))
    return windows


def _compound_is_annualized_return(
    fold_results: Sequence[WalkForwardFoldResult],
    bars_per_year: int,
) -> float | None:
    """
    Tổng hợp Annualized Return trên IS qua nhiều Fold bằng phép nhân tăng trưởng ghép thời gian:
        ((Π G_{IS, k}) ** (bars_per_year / Σ N_{IS, k}) - 1) * 100
    """
    growth = 1.0
    total_bars = 0
    for fr in fold_results:
        if fr.train_growth_factor is None or fr.train_growth_factor <= 0.0 or fr.train_return_bars <= 0:
            return None
        growth *= float(fr.train_growth_factor)
        total_bars += int(fr.train_return_bars)
    if total_bars <= 0:
        return None
    return calculate_cagr_pct(growth, n_return_bars=total_bars, bars_per_year=bars_per_year)


def run_walk_forward(
    market_data: pd.DataFrame,
    strategy_or_cls: Any,
    config: EngineConfig | None = None,
    *,
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    folds: int = 5,
    mode: WalkForwardMode | str = WalkForwardMode.ROLLING,
    train_bars: int | None = None,
    test_bars: int | None = None,
    objective_metric: str = "sharpe_annual",
    min_train_trades: int = 0,
    embargo_bars: int = 0,
    run_mode: StrategyMode | None = None,
) -> WalkForwardResult:
    """
    Walk-Forward Optimization & Validation tổng quát cho mọi Strategy (H4):
    - Tối ưu hóa hoàn toàn trên IS (dùng shared grid_runner + Strategy instance mới).
    - Independent Fold Execution cho mỗi OOS Fold (force_close_at_end=True), kèm Warm-up Prefix.
    - Loại bỏ hoàn toàn Warm-up Prefix trước khi ghép nối OOS theo thời gian (OOS Stitching).
    """
    df = _validate_time_series(market_data)
    eff_mode = mode if isinstance(mode, WalkForwardMode) else WalkForwardMode(str(mode).upper())
    if objective_metric not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective_metric '{objective_metric}' không được hỗ trợ.")
    if embargo_bars < 0:
        raise ValueError("embargo_bars phải >= 0.")

    eff_cfg = clone_config(config)
    initial_capital = float(eff_cfg.account.initial_balance)
    bars_per_year = resolve_bars_per_year(config=eff_cfg, time_series=df["time"])

    windows = _generate_fold_windows(
        len(df),
        folds=folds,
        mode=eff_mode,
        train_bars=train_bars,
        test_bars=test_bars,
        embargo_bars=embargo_bars,
    )
    if not windows:
        return WalkForwardResult(
            status=ValidationStatus.NOT_RUN,
            mode=eff_mode,
            objective_metric=objective_metric,
            messages=["NO_VALID_FOLDS: Không tạo được cửa sổ Walk-Forward phù hợp."],
        )

    fold_results: list[WalkForwardFoldResult] = []
    messages: list[str] = []

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
        train_df = df.iloc[tr_s:tr_e].copy().reset_index(drop=True)
        # Bỏ qua nếu dữ liệu IS quá ngắn cho bất kỳ warmup tối thiểu nào (grid runner tự lọc từng combo)
        grid_runs = run_parameter_grid(
            train_df,
            strategy_or_cls,
            eff_cfg,
            param_grid=param_grid,
            objective_metric=objective_metric,
            min_trades=min_train_trades,
            run_mode=run_mode,
        )
        best = select_best_run(grid_runs)
        if best is None or best.score is None:
            messages.append(
                f"FOLD_{fold_idx}_SKIPPED: Không có tổ hợp tham số hợp lệ trên IS [{tr_s}:{tr_e})."
            )
            continue

        # Lấy warmup chính xác của bộ tham số được chọn để gắn Warm-up Prefix cho OOS
        from quant_backtester.validation.grid_runner import instantiate_fresh_strategy

        selected_probe = instantiate_fresh_strategy(strategy_or_cls, best.params)
        wp = int(getattr(selected_probe, "warmup_period", 0) or 0)
        prefix_start = max(0, te_s - wp)
        oos_trade_start = te_s - prefix_start
        oos_input_df = df.iloc[prefix_start:te_e].copy().reset_index(drop=True)
        if len(oos_input_df) < wp:
            messages.append(
                f"FOLD_{fold_idx}_SKIPPED: OOS input ({len(oos_input_df)}) ngắn hơn warmup ({wp})."
            )
            continue

        oos_res = run_backtest(
            oos_input_df,
            strategy_or_cls,
            eff_cfg,
            params=best.params,
            trade_start=oos_trade_start,
            run_mode=run_mode,
            force_close_at_end=True,
        )
        oos_metrics, oos_win = compute_evaluation_metrics(oos_res)
        oos_fc = summarize_force_closes(oos_res)

        ret_series_df = compute_return_series(
            oos_win.anchored_equity_df,
            ledger_df=oos_win.evaluation_ledger_df,
            initial_balance=oos_win.anchor_equity,
        )
        # Bỏ dòng neo (index 0), chỉ lấy đúng các nến OOS thực tế [te_s .. te_e)
        oos_slice = pd.DataFrame(
            {
                "time": ret_series_df["time"].iloc[1:].reset_index(drop=True),
                "twr_return": ret_series_df["twr_return"].iloc[1:].astype(float).reset_index(drop=True),
                "fold_id": fold_idx,
            }
        )

        # Kiểm tra không overlap giữa các Fold OOS
        if fold_results and te_s <= fold_results[-1].test_end_index - 1:
            raise AssertionError("OOS folds không được chồng lấp nhau.")

        oos_gap = (
            te_s - fold_results[-1].test_end_index
            if fold_results
            else 0
        )
        fold_results.append(
            WalkForwardFoldResult(
                fold_id=fold_idx,
                mode=eff_mode.value,
                train_start_index=tr_s,
                train_end_index=tr_e,
                test_start_index=te_s,
                test_end_index=te_e,
                embargo_bars=int(te_s - tr_e),
                oos_gap_bars=int(oos_gap),
                train_start_time=pd.Timestamp(train_df["time"].iloc[0]),
                train_end_time=pd.Timestamp(train_df["time"].iloc[-1]),
                train_bars=len(train_df),
                test_start_time=pd.Timestamp(df["time"].iloc[te_s]),
                test_end_time=pd.Timestamp(df["time"].iloc[te_e - 1]),
                test_bars=te_e - te_s,
                best_params=dict(best.params),
                train_objective_score=float(best.score),
                train_metrics=dict(best.metrics),
                test_metrics=oos_metrics,
                oos_force_close_summary=oos_fc,
                oos_returns_slice=oos_slice,
                oos_trades=oos_win.evaluation_trades_df.copy(),
                train_growth_factor=best.growth_factor,
                train_return_bars=best.n_return_bars,
            )
        )

    if not fold_results:
        return WalkForwardResult(
            status=ValidationStatus.WARNING,
            mode=eff_mode,
            objective_metric=objective_metric,
            messages=messages or ["NO_COMPLETED_FOLDS: Không có Fold nào hoàn tất."],
        )

    # Ghép nối chuỗi OOS theo thứ tự thời gian (Chronological OOS Stitching)
    stitched_slices = pd.concat([f.oos_returns_slice for f in fold_results], ignore_index=True)
    stitched_times = pd.to_datetime(stitched_slices["time"])
    if not stitched_times.is_monotonic_increasing or stitched_times.duplicated().any():
        raise AssertionError("Chuỗi OOS ghép nối phải tăng dần theo thời gian và không trùng lặp.")

    twr = stitched_slices["twr_return"].to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + twr)
    eq_vals = initial_capital * wealth

    stitched_eq_for_analytics = pd.DataFrame(
        {"time": stitched_slices["time"], "equity": eq_vals}
    )
    dd_df = compute_drawdown_series(stitched_eq_for_analytics, initial_balance=initial_capital)

    stitched_oos_equity = pd.DataFrame(
        {
            "time": stitched_slices["time"],
            "equity": eq_vals,
            "twr_return": twr,
            "wealth_index": wealth,
            "drawdown_pct": dd_df["drawdown_pct"].to_numpy(dtype=float),
            "fold_id": stitched_slices["fold_id"].to_numpy(dtype=int),
        }
    )

    non_empty_trades = [f.oos_trades for f in fold_results if not f.oos_trades.empty]
    stitched_trades = (
        pd.concat(non_empty_trades, ignore_index=True)
        if non_empty_trades
        else pd.DataFrame()
    )

    # Dùng điểm neo tại t_0 - delta trước quan sát OOS đầu tiên để Analytics Group G tính đúng len(twr) quan sát
    first_t = pd.Timestamp(stitched_times.iloc[0])
    delta = (
        pd.Timestamp(stitched_times.iloc[1]) - first_t
        if len(stitched_times) >= 2
        else pd.Timedelta(nanoseconds=1)
    )
    anchored_stitched = pd.concat(
        [
            pd.DataFrame([{"time": first_t - delta, "equity": initial_capital}]),
            stitched_eq_for_analytics,
        ],
        ignore_index=True,
    )
    stitched_metrics = calculate_metrics(
        {"equity": anchored_stitched, "trades": stitched_trades},
        clone_config(eff_cfg),
    )

    # Tính WFE theo quy ước chặt chẽ
    is_ann_ret = _compound_is_annualized_return(fold_results, bars_per_year=bars_per_year)
    oos_ann_ret = stitched_metrics.get("cagr_pct")
    wfe: float | None = None
    if is_ann_ret is None or oos_ann_ret is None:
        messages.append("WFE_UNDEFINED: Không tính được Annualized Return IS hoặc OOS.")
    elif is_ann_ret <= 0.0:
        messages.append(
            f"NON_POSITIVE_IS_RETURN: IS Annualized Return ({is_ann_ret:.4f}%) <= 0, WFE trả về None."
        )
    else:
        wfe = float(oos_ann_ret) / float(is_ann_ret)

    fold_rows = []
    for f in fold_results:
        row: dict[str, Any] = {
            "fold": f.fold_id,
            "mode": f.mode,
            "train_start_time": f.train_start_time,
            "train_end_time": f.train_end_time,
            "train_bars": f.train_bars,
            "test_start_time": f.test_start_time,
            "test_end_time": f.test_end_time,
            "test_bars": f.test_bars,
            "embargo_bars": f.embargo_bars,
            "oos_gap_bars": f.oos_gap_bars,
            "best_params": dict(f.best_params),
            "train_score": f.train_objective_score,
            "train_sharpe": f.train_metrics.get("sharpe_annual", 0.0),
            "test_return_pct": f.test_metrics.get("total_return_pct", 0.0),
            "test_max_drawdown_pct": f.test_metrics.get("max_drawdown_pct", 0.0),
            "test_sharpe": f.test_metrics.get("sharpe_annual", 0.0),
            "test_trades": f.test_metrics.get("total_trades", 0),
            "oos_force_closed": f.oos_force_close_summary.closed_positions_count,
        }
        for pk, pv in f.best_params.items():
            row.setdefault(pk, pv)
        fold_rows.append(row)
    folds_df = pd.DataFrame(fold_rows)

    summary_df = pd.DataFrame(
        [
            {
                "mode": eff_mode.value,
                "completed_folds": len(fold_results),
                "combined_oos_return_pct": stitched_metrics.get("total_return_pct"),
                "combined_oos_cagr_pct": oos_ann_ret,
                "combined_oos_max_drawdown_pct": stitched_metrics.get("max_drawdown_pct"),
                "combined_oos_sharpe": stitched_metrics.get("sharpe_annual"),
                "oos_bars": len(stitched_oos_equity),
                "is_annualized_return_pct": is_ann_ret,
                "walk_forward_efficiency": wfe,
            }
        ]
    )

    status = ValidationStatus.WARNING if messages else ValidationStatus.PASS
    return WalkForwardResult(
        status=status,
        mode=eff_mode,
        objective_metric=objective_metric,
        folds=fold_results,
        folds_df=folds_df,
        summary_df=summary_df,
        stitched_oos_equity=stitched_oos_equity,
        stitched_oos_trades=stitched_trades,
        stitched_oos_metrics=stitched_metrics,
        is_annualized_return_pct=is_ann_ret,
        oos_annualized_return_pct=oos_ann_ret,
        walk_forward_efficiency=wfe,
        messages=messages,
    )
