"""H7 – Parameter Sensitivity & Stability Analysis (tái sử dụng Shared Grid Runner)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_backtester.config import EngineConfig
from quant_backtester.strategies.base import StrategyMode
from quant_backtester.validation.config import StabilityConfig
from quant_backtester.validation.grid_runner import (
    GridRunResult,
    build_param_grid,
    grid_runs_to_dataframe,
    run_parameter_grid,
    select_best_run,
)
from quant_backtester.validation.models import ValidationStatus


@dataclass
class ParameterStabilityReport:
    status: ValidationStatus
    objective_metric: str
    grid_results_df: pd.DataFrame
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    mean_score: float | None = None
    std_score: float | None = None
    median_score: float | None = None
    profitable_param_ratio: float | None = None
    n_total_combos: int = 0
    n_valid_combos: int = 0
    n_execution_errors: int = 0
    neighbors_count: int = 0
    neighborhood_mean_score: float | None = None
    neighborhood_median_score: float | None = None
    neighborhood_score_std: float | None = None
    neighborhood_score_gap: float | None = None
    neighborhood_retention_ratio: float | None = None
    metric_coefficient_of_variation: float | None = None
    is_isolated_peak: bool = False
    is_stable_peak: bool = False
    surface_matrix: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)


def _param_key(params: Mapping[str, Any], axis_names: Sequence[str]) -> tuple[Any, ...]:
    return tuple(params.get(name) for name in axis_names)


def _compute_neighborhood_scores(
    runs: Sequence[GridRunResult],
    best: GridRunResult,
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None,
) -> list[float]:
    """
    Tìm các điểm lân cận bậc 1 trên lưới rời rạc đã sắp xếp theo từng trục:
        max_k |i_k - j_k| == 1
    KHÔNG bao gồm chính điểm trung tâm.
    """
    axis_names = (
        list(param_grid.keys())
        if isinstance(param_grid, Mapping)
        else list(best.params.keys())
    )
    if not axis_names:
        return []

    axis_values: dict[str, list[Any]] = {}
    for name in axis_names:
        raw_vals = []
        for r in runs:
            v = r.params.get(name)
            if v not in raw_vals:
                raw_vals.append(v)
        try:
            axis_values[name] = sorted(raw_vals)
        except TypeError:
            axis_values[name] = raw_vals

    coord_map: dict[tuple[Any, ...], tuple[int, ...]] = {}
    for name_vals in [axis_values]:
        for r in runs:
            k = _param_key(r.params, axis_names)
            coord_map[k] = tuple(name_vals[n].index(r.params.get(n)) for n in axis_names)

    best_coord = coord_map.get(_param_key(best.params, axis_names))
    if best_coord is None:
        return []

    neighbor_scores: list[float] = []
    for r in runs:
        if not r.is_valid or r.score is None:
            continue
        k = _param_key(r.params, axis_names)
        c = coord_map.get(k)
        if c is None or c == best_coord:
            continue
        chebyshev = max(abs(a - b) for a, b in zip(c, best_coord))
        if chebyshev == 1:
            neighbor_scores.append(float(r.score))
    return neighbor_scores


def analyze_parameter_stability(
    market_data: pd.DataFrame,
    strategy_or_cls: Any,
    config: EngineConfig | None = None,
    *,
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    objective_metric: str = "sharpe_annual",
    min_trades: int = 0,
    stability_config: StabilityConfig | None = None,
    run_mode: StrategyMode | None = None,
) -> ParameterStabilityReport:
    """
    Quét lưới tham số (H7) trên tập dữ liệu phân tích (IS / Validation), không dùng OOS để chọn tham số.
    Đánh giá đồng thời nhiều chỉ số: Profitable Ratio, Neighborhood Mean/Median/Std/Gap, CV và Isolated Peak.
    """
    if market_data is None or market_data.empty:
        return ParameterStabilityReport(
            status=ValidationStatus.NOT_RUN,
            objective_metric=objective_metric,
            grid_results_df=pd.DataFrame(),
            warnings=["EMPTY_DATA: Không có dữ liệu để kiểm tra ổn định tham số."],
        )

    scfg = stability_config if stability_config is not None else StabilityConfig()
    runs = run_parameter_grid(
        market_data,
        strategy_or_cls,
        config=config,
        param_grid=param_grid,
        objective_metric=objective_metric,
        min_trades=min_trades,
        run_mode=run_mode,
    )
    df = grid_runs_to_dataframe(runs)
    n_total = len(runs)
    n_errors = sum(1 for r in runs if r.invalid_reason == "EXECUTION_ERROR")
    valid_runs = [r for r in runs if r.is_valid and r.score is not None]

    # Profitable Ratio không tính các tổ hợp lỗi thực thi vào mẫu số hay tử số
    executed_ok = [
        r for r in runs
        if r.invalid_reason != "EXECUTION_ERROR" and r.total_return_pct is not None and math.isfinite(float(r.total_return_pct))
    ]
    prof_ratio = (
        sum(1 for r in executed_ok if float(r.total_return_pct) > 0.0) / float(len(executed_ok))
        if executed_ok
        else None
    )

    best = select_best_run(runs)
    if best is None or best.score is None:
        return ParameterStabilityReport(
            status=ValidationStatus.WARNING,
            objective_metric=objective_metric,
            grid_results_df=df,
            profitable_param_ratio=prof_ratio,
            n_total_combos=n_total,
            n_valid_combos=len(valid_runs),
            n_execution_errors=n_errors,
            warnings=["NO_VALID_PARAMETERS: Không có tổ hợp tham số nào hợp lệ."],
        )

    scores = np.array([float(r.score) for r in valid_runs], dtype=float)
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores, ddof=1)) if len(scores) >= 2 else 0.0
    med_s = float(np.median(scores))
    cv = (std_s / abs(mean_s)) if abs(mean_s) > 1e-12 else None

    # Pivot surface 2D nếu lưới có đúng 2 tham số
    surface: pd.DataFrame | None = None
    inferred_axes = (
        list(param_grid.keys())
        if isinstance(param_grid, Mapping)
        else [k for k in best.params.keys()]
    )
    if len(inferred_axes) == 2 and all(a in df.columns for a in inferred_axes):
        surface = df.pivot_table(
            index=inferred_axes[0],
            columns=inferred_axes[1],
            values="score",
            aggfunc="mean",
        )

    neighbor_scores = _compute_neighborhood_scores(runs, best, param_grid)
    nb_count = len(neighbor_scores)
    warnings: list[str] = []

    if nb_count > 0:
        nb_arr = np.array(neighbor_scores, dtype=float)
        nb_mean = float(np.mean(nb_arr))
        nb_med = float(np.median(nb_arr))
        nb_std = float(np.std(nb_arr, ddof=1)) if nb_count >= 2 else 0.0
        nb_gap = float(best.score - nb_mean)
        # Chỉ tính retention ratio khi best_score > 0 (tránh chia cho 0 hoặc âm gây đảo chiều)
        retention = (nb_mean / float(best.score)) if float(best.score) > 1e-12 else None
    else:
        nb_mean = None
        nb_med = None
        nb_std = None
        nb_gap = None
        retention = None
        warnings.append("NO_NEIGHBORS: Lưới chỉ có 1 điểm hoặc không có lân cận hợp lệ.")

    isolated = False
    if nb_count >= scfg.min_neighbors and nb_mean is not None:
        best_pos = float(best.score)
        rel_gap = ((best_pos - nb_mean) / abs(best_pos)) if abs(best_pos) > 1e-12 else 0.0
        sharp_drop = (rel_gap > scfg.max_relative_gap) or (nb_mean <= scfg.neighborhood_floor and best_pos > scfg.neighborhood_floor)
        low_prof = prof_ratio is not None and prof_ratio < scfg.min_profitable_ratio
        high_disp = nb_std is not None and abs(nb_mean) > 1e-12 and (nb_std / abs(nb_mean) > scfg.max_neighbor_dispersion)
        if sharp_drop and (low_prof or high_disp or nb_mean <= scfg.neighborhood_floor):
            isolated = True
            warnings.append(
                f"ISOLATED_PEAK_DETECTED: best_score={best_pos:.4f}, neighbor_mean={nb_mean:.4f}, "
                f"profitable_ratio={prof_ratio if prof_ratio is not None else 'N/A'}."
            )

    if prof_ratio is not None and prof_ratio < scfg.min_profitable_ratio:
        warnings.append(
            f"LOW_PROFITABLE_PARAM_RATIO: {prof_ratio:.2%} < ngưỡng {scfg.min_profitable_ratio:.2%}."
        )

    is_stable = (
        not isolated
        and (prof_ratio is None or prof_ratio >= scfg.min_profitable_ratio)
        and (nb_count >= scfg.min_neighbors or n_total == 1)
    )
    status = ValidationStatus.WARNING if (isolated or warnings) else ValidationStatus.PASS

    return ParameterStabilityReport(
        status=status,
        objective_metric=objective_metric,
        grid_results_df=df,
        best_params=dict(best.params),
        best_score=float(best.score),
        mean_score=mean_s,
        std_score=std_s,
        median_score=med_s,
        profitable_param_ratio=prof_ratio,
        n_total_combos=n_total,
        n_valid_combos=len(valid_runs),
        n_execution_errors=n_errors,
        neighbors_count=nb_count,
        neighborhood_mean_score=nb_mean,
        neighborhood_median_score=nb_med,
        neighborhood_score_std=nb_std,
        neighborhood_score_gap=nb_gap,
        neighborhood_retention_ratio=retention,
        metric_coefficient_of_variation=cv,
        is_isolated_peak=isolated,
        is_stable_peak=is_stable,
        surface_matrix=surface,
        warnings=warnings,
    )
