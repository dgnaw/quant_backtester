"""H5 – Determinism, Seed Management, Input/Result Fingerprint & Reproducibility Check."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from typing import Any, Iterator

import numpy as np
import pandas as pd

from quant_backtester.config import EngineConfig
from quant_backtester.validation.config import ValidationConfig
from quant_backtester.validation.grid_runner import (
    instantiate_fresh_strategy,
    run_backtest,
    strategy_parameters_of,
)
from quant_backtester.validation.models import ValidationStatus

ENGINE_VERSION = "1.0.0"
FINGERPRINT_SCHEMA_VERSION = "1.0"


def make_rng(seed: int) -> np.random.Generator:
    """Tạo đối tượng numpy.random.Generator riêng biệt, không chạm global state."""
    return np.random.default_rng(int(seed))


@contextmanager
def deterministic_seed_context(seed: int) -> Iterator[np.random.Generator]:
    """
    Quản lý seed cho random, numpy.random và trả về numpy.random.Generator riêng.
    Lưu và khôi phục hoàn toàn global RNG state sau khi thoát để không gây side effect.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        yield make_rng(int(seed))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _normalize_scalar(val: Any, precision: int) -> Any:
    if val is None or isinstance(val, (str, bool)):
        return val
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            return "__NaT__"
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(val, pd.Timedelta):
        return f"TD:{val.total_seconds():.9f}"
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if math.isnan(f):
            return "__NaN__"
        if math.isinf(f):
            return "__+INF__" if f > 0 else "__-INF__"
        rounded = round(f, precision)
        if rounded == 0.0:  # chuẩn hóa -0.0 -> 0.0
            return 0.0
        return rounded
    if pd.isna(val):
        return "__NaN__"
    if isinstance(val, dict):
        return {str(k): _normalize_scalar(v, precision) for k, v in sorted(val.items(), key=lambda x: str(x[0]))}
    if isinstance(val, (list, tuple)):
        return [_normalize_scalar(v, precision) for v in val]
    if hasattr(val, "value"):  # Enum
        return _normalize_scalar(val.value, precision)
    if hasattr(val, "__dict__"):
        return _normalize_scalar(vars(val), precision)
    return str(val)


def _normalize_dataframe(
    df: pd.DataFrame | None,
    precision: int,
    sort_cols: tuple[str, ...] = ("time", "timestamp", "order_id", "entry_id", "trade_id"),
) -> list[dict[str, Any]]:
    """Chuẩn hóa DataFrame để băm: không phụ thuộc thứ tự cột, chuẩn NaN/Inf/Timezone/Float."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    work = df.copy()
    sorted_cols = sorted(work.columns.tolist())
    work = work[sorted_cols]
    present_sort = [c for c in sort_cols if c in work.columns]
    if present_sort:
        work = work.sort_values(by=present_sort, kind="mergesort")
    work = work.reset_index(drop=True)
    records = work.to_dict(orient="records")
    return [
        {str(col): _normalize_scalar(rec[col], precision) for col in sorted_cols}
        for rec in records
    ]


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_input_fingerprint(
    market_data: pd.DataFrame,
    strategy_or_cls: Any,
    config: EngineConfig | None = None,
    validation_config: ValidationConfig | None = None,
    *,
    seed: int | None = None,
    precision: int | None = None,
) -> str:
    """Tính mã băm SHA-256 cho toàn bộ đầu vào của một lần chạy Backtest."""
    vcfg = validation_config if validation_config is not None else ValidationConfig()
    eff_seed = vcfg.random_seed if seed is None else int(seed)
    eff_prec = vcfg.fingerprint_precision if precision is None else int(precision)
    cfg = config if config is not None else EngineConfig()

    cls = strategy_or_cls if isinstance(strategy_or_cls, type) else strategy_or_cls.__class__
    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "seed": eff_seed,
        "strategy_class": f"{cls.__module__}.{cls.__qualname__}",
        "strategy_parameters": _normalize_scalar(strategy_parameters_of(strategy_or_cls), eff_prec),
        "engine_config": _normalize_scalar(cfg, eff_prec),
        "validation_config": _normalize_scalar(vcfg, eff_prec),
        "market_data": _normalize_dataframe(market_data, eff_prec),
    }
    return _sha256_json(payload)


def _extract_fills(result: Any) -> pd.DataFrame:
    ev = getattr(result, "order_events", None)
    if not isinstance(ev, pd.DataFrame) or ev.empty or "event_type" not in ev.columns:
        return pd.DataFrame()
    return ev[ev["event_type"] == "FILLED"].copy()


def compute_result_fingerprint(
    result: Any,
    precision: int = 10,
) -> str:
    """
    Tính mã băm SHA-256 cho toàn bộ kết quả đầu ra đã chuẩn hóa:
    Equity Curve, Orders, Fills, Trades, Ledger và Metrics.
    """
    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "equity": _normalize_dataframe(getattr(result, "equity", None), precision),
        "orders": _normalize_dataframe(getattr(result, "orders", None), precision),
        "fills": _normalize_dataframe(_extract_fills(result), precision),
        "trades": _normalize_dataframe(getattr(result, "trades", None), precision),
        "ledger": _normalize_dataframe(getattr(result, "ledger", None), precision),
        "metrics": _normalize_scalar(getattr(result, "metrics", {}) or {}, precision),
    }
    return _sha256_json(payload)


def _diff_components(ref: Any, cur: Any, precision: int, run_idx: int) -> list[str]:
    diffs: list[str] = []
    for name in ("equity", "orders", "trades", "ledger"):
        a = _normalize_dataframe(getattr(ref, name, None), precision)
        b = _normalize_dataframe(getattr(cur, name, None), precision)
        if a != b:
            diffs.append(f"Run #{run_idx}: bảng '{name}' khác biệt với Run #0.")
    if _normalize_dataframe(_extract_fills(ref), precision) != _normalize_dataframe(_extract_fills(cur), precision):
        diffs.append(f"Run #{run_idx}: danh sách Fills khác biệt với Run #0.")
    if _normalize_scalar(getattr(ref, "metrics", {}), precision) != _normalize_scalar(getattr(cur, "metrics", {}), precision):
        diffs.append(f"Run #{run_idx}: metrics khác biệt với Run #0.")
    return diffs


@dataclass
class ReproducibilityReport:
    status: ValidationStatus
    seed: int
    n_runs: int
    input_fingerprints: list[str] = field(default_factory=list)
    result_fingerprints: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    classification: str = ""

    @property
    def is_reproducible(self) -> bool:
        return self.status == ValidationStatus.PASS


def classify_fingerprint_pair(
    input_fp_a: str,
    input_fp_b: str,
    result_fp_a: str,
    result_fp_b: str,
) -> str:
    same_in = input_fp_a == input_fp_b
    same_out = result_fp_a == result_fp_b
    if same_in and same_out:
        return "SAME_INPUT_SAME_RESULT"
    if same_in and not same_out:
        return "SAME_INPUT_DIFFERENT_RESULT"
    if not same_in and same_out:
        return "DIFFERENT_INPUT_SAME_RESULT"
    return "DIFFERENT_INPUT_DIFFERENT_RESULT"


def verify_reproducibility(
    market_data: pd.DataFrame,
    strategy: Any,
    config: EngineConfig | None = None,
    validation_config: ValidationConfig | None = None,
    *,
    seed: int | None = None,
    n_runs: int | None = None,
    trade_start: int = 0,
) -> ReproducibilityReport:
    """
    Chạy lặp lại Backtest n_runs lần với cùng (market_data, config, seed) và Strategy instance MỚI mỗi lần.
    Xác minh cả Input Fingerprint và Result Fingerprint (Equity, Orders, Fills, Trades, Ledger, Metrics).
    """
    if market_data is None or market_data.empty:
        vcfg = validation_config if validation_config is not None else ValidationConfig()
        return ReproducibilityReport(
            status=ValidationStatus.NOT_RUN,
            seed=vcfg.random_seed if seed is None else int(seed),
            n_runs=0,
            differences=["EMPTY_DATA: Không có dữ liệu thị trường."],
        )

    vcfg = validation_config if validation_config is not None else ValidationConfig()
    eff_seed = vcfg.random_seed if seed is None else int(seed)
    eff_runs = vcfg.reproducibility_runs if n_runs is None else max(2, int(n_runs))
    precision = vcfg.fingerprint_precision

    input_fps: list[str] = []
    result_fps: list[str] = []
    results: list[Any] = []
    differences: list[str] = []

    for idx in range(eff_runs):
        fresh_strategy = instantiate_fresh_strategy(strategy)
        in_fp = compute_input_fingerprint(
            market_data, fresh_strategy, config, vcfg, seed=eff_seed, precision=precision
        )
        with deterministic_seed_context(eff_seed) as rng:
            if hasattr(fresh_strategy, "set_rng"):
                fresh_strategy.set_rng(rng)
            elif hasattr(fresh_strategy, "rng"):
                fresh_strategy.rng = rng
            res = run_backtest(market_data, fresh_strategy, config=config, trade_start=trade_start)
        out_fp = compute_result_fingerprint(res, precision=precision)
        input_fps.append(in_fp)
        result_fps.append(out_fp)
        results.append(res)
        if idx > 0 and out_fp != result_fps[0]:
            differences.extend(_diff_components(results[0], res, precision, idx))

    unique_in = set(input_fps)
    unique_out = set(result_fps)
    classification = classify_fingerprint_pair(
        input_fps[0], input_fps[-1], result_fps[0], result_fps[-1]
    )
    if len(unique_in) > 1:
        differences.append("INPUT_FINGERPRINT_DRIFT: Input Fingerprint không đồng nhất giữa các lần chạy.")

    status = ValidationStatus.PASS if (len(unique_in) == 1 and len(unique_out) == 1) else ValidationStatus.FAIL
    return ReproducibilityReport(
        status=status,
        seed=eff_seed,
        n_runs=eff_runs,
        input_fingerprints=input_fps,
        result_fingerprints=result_fps,
        differences=differences,
        classification=classification,
    )
