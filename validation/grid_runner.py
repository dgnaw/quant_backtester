"""Shared Parameter Grid Runner (DH5) và tiện ích cô lập Strategy instance (DH4)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import product
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_backtester.config import EngineConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.result import BacktestResult
from quant_backtester.strategies.base import Strategy, StrategyMode

SUPPORTED_OBJECTIVES = (
    "sharpe_annual",
    "sortino_annual",
    "calmar_ratio",
    "total_return_pct",
    "profit_factor",
    "sqn",
)


# ---------------------------------------------------------------------------
# Strategy isolation
# ---------------------------------------------------------------------------
def resolve_strategy_class(strategy_or_cls: Any) -> type[Strategy]:
    if isinstance(strategy_or_cls, type):
        if not issubclass(strategy_or_cls, Strategy):
            raise TypeError(f"{strategy_or_cls!r} không phải subclass của Strategy.")
        return strategy_or_cls
    if isinstance(strategy_or_cls, Strategy):
        return strategy_or_cls.__class__
    raise TypeError("strategy_or_cls phải là Strategy class hoặc Strategy instance.")


def instantiate_fresh_strategy(strategy_or_cls: Any, params: Mapping[str, Any] | None = None) -> Strategy:
    """
    Tạo một Strategy instance MỚI, độc lập (DH4).
    - Class: cls(**params).
    - Instance: cls(**(instance.get_parameters() | params)); nếu class không nhận kwargs
      (ví dụ strategy kịch bản), fallback sang deepcopy + reset(). Không bao giờ trả về object gốc.
    """
    overrides = dict(params or {})
    if isinstance(strategy_or_cls, type):
        cls = resolve_strategy_class(strategy_or_cls)
        return cls(**overrides)

    cls = resolve_strategy_class(strategy_or_cls)
    merged = dict(strategy_or_cls.get_parameters())
    merged.update(overrides)
    try:
        fresh = cls(**merged)
    except TypeError:
        if overrides:
            raise
        fresh = copy.deepcopy(strategy_or_cls)
    if fresh is strategy_or_cls:  # pragma: no cover - phòng vệ
        fresh = copy.deepcopy(strategy_or_cls)
    if hasattr(fresh, "reset"):
        fresh.reset()
    return fresh


def strategy_parameters_of(strategy_or_cls: Any, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(strategy_or_cls, Strategy):
        merged = dict(strategy_or_cls.get_parameters())
        merged.update(dict(params or {}))
        return merged
    return dict(params or {})


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
def _schema_values(param: Any, max_points: int) -> list[Any]:
    if param.choices is not None:
        return list(param.choices)
    if param.param_type is bool:
        return [False, True]
    if param.min_value is None or param.max_value is None:
        return [param.default] if param.default is not None else []
    lo, hi = float(param.min_value), float(param.max_value)
    if param.step is not None and float(param.step) > 0:
        n_steps = int(math.floor((hi - lo) / float(param.step) + 1e-9)) + 1
        if n_steps <= max_points:
            vals = [lo + k * float(param.step) for k in range(n_steps)]
        else:
            vals = list(np.linspace(lo, hi, max_points))
    else:
        vals = list(np.linspace(lo, hi, max_points))
    if param.param_type is int:
        out: list[Any] = []
        for v in vals:
            iv = int(round(v))
            if iv not in out:
                out.append(iv)
        return out
    return [float(v) for v in vals]


def build_param_grid(
    strategy_or_cls: Any,
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    max_points_per_param: int = 5,
) -> list[dict[str, Any]]:
    """
    Tạo danh sách tổ hợp tham số theo thứ tự xác định:
    - dict[name, values]: tích Descartes theo thứ tự khóa và thứ tự giá trị đã cho.
    - list[dict]: dùng nguyên trạng.
    - None: suy ra từ Parameter(optimizable=True) trong Strategy.parameters; nếu không có, trả về [{}].
    """
    if param_grid is not None:
        if isinstance(param_grid, Mapping):
            names = list(param_grid.keys())
            values = [list(param_grid[n]) for n in names]
            if any(len(v) == 0 for v in values):
                raise ValueError("Mỗi tham số trong param_grid phải có ít nhất một giá trị.")
            return [dict(zip(names, combo)) for combo in product(*values)]
        combos = [dict(c) for c in param_grid]
        if not combos:
            raise ValueError("param_grid rỗng.")
        return combos

    cls = resolve_strategy_class(strategy_or_cls)
    optimizable = [p for p in getattr(cls, "parameters", ()) if getattr(p, "optimizable", False)]
    if not optimizable:
        return [{}]
    names = [p.name for p in optimizable]
    values = [_schema_values(p, max_points_per_param) for p in optimizable]
    if any(len(v) == 0 for v in values):
        return [{}]
    return [dict(zip(names, combo)) for combo in product(*values)]


# ---------------------------------------------------------------------------
# Running backtests
# ---------------------------------------------------------------------------
def clone_config(config: EngineConfig | None, **overrides: Any) -> EngineConfig:
    cfg = copy.deepcopy(config) if config is not None else EngineConfig()
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def run_backtest(
    market_data: pd.DataFrame,
    strategy_or_cls: Any,
    config: EngineConfig | None = None,
    params: Mapping[str, Any] | None = None,
    trade_start: int = 0,
    run_mode: StrategyMode | None = None,
    benchmark_data: Any = None,
    force_close_at_end: bool | None = None,
) -> BacktestResult:
    """Chạy một Backtest độc lập: Strategy mới, EngineConfig bản sao, Engine mới."""
    overrides = {} if force_close_at_end is None else {"force_close_at_end": bool(force_close_at_end)}
    cfg = clone_config(config, **overrides)
    strategy = instantiate_fresh_strategy(strategy_or_cls, params)
    engine = BacktestEngine(cfg)
    return engine.run(
        market_data.copy(),
        strategy,
        trade_start=trade_start,
        run_mode=run_mode,
        benchmark_data=benchmark_data,
    )


def extract_score(metrics: Mapping[str, Any] | None, objective_metric: str) -> float | None:
    if objective_metric not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective_metric '{objective_metric}' không được hỗ trợ. Hỗ trợ: {SUPPORTED_OBJECTIVES}.")
    if not metrics:
        return None
    if objective_metric == "sharpe_annual" and metrics.get("sharpe_is_valid") is False:
        return None
    val = metrics.get(objective_metric)
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclass
class GridRunResult:
    run_index: int
    params: dict[str, Any]
    score: float | None
    metrics: dict[str, Any] = field(default_factory=dict)
    total_trades: int = 0
    total_return_pct: float | None = None
    growth_factor: float | None = None
    n_return_bars: int = 0
    is_valid: bool = False
    invalid_reason: str | None = None
    error: str | None = None
    result: BacktestResult | None = None


def run_parameter_grid(
    market_data: pd.DataFrame,
    strategy_or_cls: Any,
    config: EngineConfig | None = None,
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    objective_metric: str = "sharpe_annual",
    min_trades: int = 0,
    trade_start: int = 0,
    run_mode: StrategyMode | None = None,
    keep_results: bool = False,
) -> list[GridRunResult]:
    """
    Chạy toàn bộ Parameter Grid (dùng chung cho Walk-Forward và Parameter Stability).
    Điểm số được tính trên Evaluation Window (loại bỏ vùng warm-up), dùng Analytics Group G.
    Lỗi thực thi được ghi nhận là invalid, KHÔNG bị coi là kết quả lỗ.
    """
    from quant_backtester.validation.warmup_guard import compute_evaluation_metrics

    if objective_metric not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective_metric '{objective_metric}' không được hỗ trợ.")
    combos = build_param_grid(strategy_or_cls, param_grid)
    runs: list[GridRunResult] = []
    for idx, combo in enumerate(combos):
        try:
            res = run_backtest(market_data, strategy_or_cls, config, combo, trade_start=trade_start, run_mode=run_mode)
            eval_metrics, window = compute_evaluation_metrics(res)
        except Exception as exc:  # noqa: BLE001 - ghi nhận mọi lỗi thực thi
            runs.append(
                GridRunResult(
                    run_index=idx,
                    params=dict(combo),
                    score=None,
                    is_valid=False,
                    invalid_reason="EXECUTION_ERROR",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        score = extract_score(eval_metrics, objective_metric)
        trades = int(eval_metrics.get("total_trades", 0) or 0)
        reason = None
        if score is None:
            reason = "INVALID_OBJECTIVE"
        elif trades < int(min_trades):
            reason = "INSUFFICIENT_TRADES"
        anchor = window.anchor_equity
        final_eq = float(window.anchored_equity_df["equity"].iloc[-1])
        growth = final_eq / anchor if anchor > 0 else None
        runs.append(
            GridRunResult(
                run_index=idx,
                params=dict(combo),
                score=score,
                metrics=eval_metrics,
                total_trades=trades,
                total_return_pct=eval_metrics.get("total_return_pct"),
                growth_factor=growth,
                n_return_bars=window.return_observations,
                is_valid=reason is None,
                invalid_reason=reason,
                result=res if keep_results else None,
            )
        )
    return runs


def select_best_run(runs: Iterable[GridRunResult]) -> GridRunResult | None:
    """Chọn run hợp lệ có score cao nhất; tie-break theo thứ tự xuất hiện trong Grid."""
    best: GridRunResult | None = None
    for run in runs:
        if not run.is_valid or run.score is None:
            continue
        if best is None or run.score > best.score:  # '>' giữ run xuất hiện trước khi bằng điểm
            best = run
    return best


def grid_runs_to_dataframe(runs: Sequence[GridRunResult]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row: dict[str, Any] = {"run_index": r.run_index}
        row.update(r.params)
        row.update(
            {
                "params": dict(r.params),
                "score": r.score,
                "total_trades": r.total_trades,
                "total_return_pct": r.total_return_pct,
                "is_valid": r.is_valid,
                "invalid_reason": r.invalid_reason,
                "error": r.error,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
