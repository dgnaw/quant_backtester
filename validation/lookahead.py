"""H1 – Anti-Lookahead Verification (Prefix Truncation, Future Perturbation, Order Eligibility & Prefix State Equivalence)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_backtester.config import EngineConfig
from quant_backtester.data.validator import DataValidator
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.order import OrderOrigin
from quant_backtester.strategies.adapters import frame_to_signals
from quant_backtester.strategies.base import StrategyMode
from quant_backtester.validation.config import ValidationConfig
from quant_backtester.validation.grid_runner import (
    clone_config,
    instantiate_fresh_strategy,
)
from quant_backtester.validation.models import ValidationStatus, worst_status

_COMPARE_TOL = 1e-9
_NON_NEXT_BAR_ORIGINS = {
    OrderOrigin.END_OF_TEST.value,
    OrderOrigin.PROTECTIVE_SL.value,
    OrderOrigin.PROTECTIVE_TP.value,
}


@dataclass(frozen=True)
class LookaheadViolation:
    bar_index: int
    time: pd.Timestamp | None
    check_type: str  # PREFIX_SIGNAL | FUTURE_PERTURBATION | ORDER_ELIGIBILITY | PREFIX_BACKTEST
    source: str      # STRATEGY | ENGINE
    details: str


@dataclass
class LookaheadAuditReport:
    status: ValidationStatus
    strategy_status: ValidationStatus
    engine_status: ValidationStatus
    checked_bars: list[int] = field(default_factory=list)
    violations: list[LookaheadViolation] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.FAIL

    @property
    def strategy_violations(self) -> list[LookaheadViolation]:
        return [v for v in self.violations if v.source == "STRATEGY"]

    @property
    def engine_violations(self) -> list[LookaheadViolation]:
        return [v for v in self.violations if v.source == "ENGINE"]


def _resolve_sample_bars(
    n_bars: int,
    warmup: int,
    *,
    check_all_bars: bool = False,
    sample_indices: Sequence[int] | None = None,
    sample_step: int = 10,
    max_samples: int | None = 20,
    random_seed: int = 42,
) -> list[int]:
    min_bar = max(0, int(warmup))
    max_bar = n_bars - 2  # Cần ít nhất 1 nến tương lai để cắt ngắn / perturb
    if min_bar > max_bar:
        return []

    if sample_indices is not None:
        return sorted({int(i) for i in sample_indices if min_bar <= int(i) <= max_bar})

    candidates = list(range(min_bar, max_bar + 1))
    if check_all_bars:
        return candidates

    step = max(1, int(sample_step))
    sampled = list(range(min_bar, max_bar + 1, step))
    if max_bar not in sampled:
        sampled.append(max_bar)
    if max_samples is not None and len(sampled) > int(max_samples):
        rng = np.random.default_rng(int(random_seed))
        core = [min_bar, max_bar]
        middle = [idx for idx in sampled if idx not in core]
        need = max(0, int(max_samples) - len(core))
        chosen = rng.choice(middle, size=min(need, len(middle)), replace=False).tolist()
        sampled = sorted(set(core + chosen))
    return sampled


def _perturb_future_bars(
    market_data: pd.DataFrame,
    split_bar: int,
    factor: float,
) -> pd.DataFrame:
    """
    Nhân đồng thời các cột OHLC ở tương lai (> split_bar) với cùng hệ số dương `factor`.
    Phép scale đồng nhất bảo toàn tuyệt đối thứ tự Low <= Open, Close <= High và giá > 0.
    """
    perturbed = market_data.copy()
    if split_bar + 1 >= len(perturbed):
        return perturbed
    ohlc_cols = [c for c in ("open", "high", "low", "close") if c in perturbed.columns]
    for col in ohlc_cols:
        perturbed.loc[split_bar + 1 :, col] = perturbed.loc[split_bar + 1 :, col].astype(float) * float(factor)
    return perturbed


def _signals_equal(sig_a: Any, sig_b: Any) -> bool:
    if sig_a.signal_type != sig_b.signal_type:
        return False
    if not math.isclose(float(sig_a.strength), float(sig_b.strength), rel_tol=1e-9, abs_tol=1e-9):
        return False
    meta_a = dict(sig_a.metadata.get("directives", {})) if sig_a.metadata else {}
    meta_b = dict(sig_b.metadata.get("directives", {})) if sig_b.metadata else {}
    if meta_a != meta_b:
        return False
    flags_a = dict(sig_a.metadata.get("legacy_flags", {})) if sig_a.metadata else {}
    flags_b = dict(sig_b.metadata.get("legacy_flags", {})) if sig_b.metadata else {}
    return flags_a == flags_b


def _extract_vectorized_signals(
    market_data: pd.DataFrame,
    strategy_or_cls: Any,
    symbol: str,
) -> list[Any]:
    strat = instantiate_fresh_strategy(strategy_or_cls)
    if hasattr(strat, "reset"):
        strat.reset()
    raw = strat.generate_signals(market_data.copy())
    return frame_to_signals(raw, symbol=symbol)


def audit_order_eligibility(result: Any) -> list[LookaheadViolation]:
    """
    Kiểm tra trực tiếp bất biến Order Eligibility từ BacktestResult:
    Lệnh sinh ra tại đóng nến t (STRATEGY_ENTRY / STRATEGY_EXIT) không được phép khớp ở bar <= t.
    """
    violations: list[LookaheadViolation] = []
    events = getattr(result, "order_events", None)
    orders = getattr(result, "orders", None)
    if not isinstance(events, pd.DataFrame) or events.empty:
        return violations

    origin_map: dict[str, str] = {}
    if isinstance(orders, pd.DataFrame) and not orders.empty and "order_id" in orders.columns:
        origin_map = dict(zip(orders["order_id"].astype(str), orders["origin"].astype(str)))

    created = events[events["event_type"] == "CREATED"]
    created_bar_by_id = {
        str(r["order_id"]): int(r["bar_index"])
        for _, r in created.iterrows()
        if pd.notna(r.get("bar_index"))
    }
    filled = events[events["event_type"] == "FILLED"]
    for _, r in filled.iterrows():
        oid = str(r["order_id"])
        origin = origin_map.get(oid, "")
        if origin in _NON_NEXT_BAR_ORIGINS:
            continue
        fill_bar = r.get("bar_index")
        c_bar = created_bar_by_id.get(oid)
        if c_bar is None or pd.isna(fill_bar):
            continue
        f_bar = int(fill_bar)
        if f_bar < c_bar + 1:
            violations.append(
                LookaheadViolation(
                    bar_index=f_bar,
                    time=pd.Timestamp(r["timestamp"]) if pd.notna(r.get("timestamp")) else None,
                    check_type="ORDER_ELIGIBILITY",
                    source="ENGINE",
                    details=(
                        f"Order {oid} ({origin}) tạo tại bar {c_bar} nhưng khớp tại bar {f_bar} (< {c_bar + 1})."
                    ),
                )
            )
    return violations


def _compare_prefix_backtest_state(
    full_res: Any,
    prefix_res: Any,
    t_bar: int,
    t_time: pd.Timestamp,
    *,
    is_full_length_perturbed: bool = False,
) -> list[LookaheadViolation]:
    """
    So sánh trạng thái Backtest tới nến t_bar.
    - Với prefix cắt ngắn (len == t_bar + 1): tại bar cuối t_bar Engine không phát tín hiệu /
      tạo order cho bar t_bar + 1, nên signal_log và orders so sánh nghiêm ngặt trước t_time (< t_time),
      còn equity, balance, position, fills và ledger so sánh tới hết bar t_bar (<= t_time).
    - Với Future Perturbation (len == N): so sánh cả signal_log và orders tới <= t_time.
    - Orders chỉ so sánh thuộc tính tạo lệnh (order_id, time, side, order_type, quantity, origin),
      không so sánh trạng thái khớp ở tương lai (> t_bar).
    """
    violations: list[LookaheadViolation] = []
    check_kind = "FUTURE_PERTURBATION" if is_full_length_perturbed else "PREFIX_BACKTEST"

    # 1. Signal log
    if is_full_length_perturbed:
        full_sigs = [s for s in full_res.signal_log if pd.Timestamp(s["timestamp"]) <= t_time]
        pref_sigs = [s for s in prefix_res.signal_log if pd.Timestamp(s["timestamp"]) <= t_time]
    else:
        full_sigs = [s for s in full_res.signal_log if pd.Timestamp(s["timestamp"]) < t_time]
        pref_sigs = [s for s in prefix_res.signal_log if pd.Timestamp(s["timestamp"]) < t_time]
    if len(full_sigs) != len(pref_sigs) or any(
        (a.get("signal_type"), a.get("decision")) != (b.get("signal_type"), b.get("decision"))
        for a, b in zip(full_sigs, pref_sigs)
    ):
        violations.append(
            LookaheadViolation(
                bar_index=t_bar,
                time=t_time,
                check_type="FUTURE_PERTURBATION" if is_full_length_perturbed else "PREFIX_SIGNAL",
                source="STRATEGY",
                details=f"Signal log tới bar {t_bar} khác nhau ({check_kind}).",
            )
        )

    # 2. Equity / Balance / Position đến bar t
    eq_full = full_res.equity.iloc[: t_bar + 1].reset_index(drop=True)
    eq_pref = prefix_res.equity.iloc[: t_bar + 1].reset_index(drop=True)
    for col in ("balance", "equity", "position"):
        if col in eq_full.columns and col in eq_pref.columns:
            diff = np.max(np.abs(eq_full[col].astype(float).to_numpy() - eq_pref[col].astype(float).to_numpy()))
            if diff > _COMPARE_TOL:
                violations.append(
                    LookaheadViolation(
                        bar_index=t_bar,
                        time=t_time,
                        check_type=check_kind,
                        source="ENGINE",
                        details=f"Cột equity['{col}'] sai lệch tối đa {diff:.6g} tại [0..{t_bar}].",
                    )
                )
                break

    # 3. Orders tạo lệnh trước t_time (hoặc <= t_time nếu là perturbed full-length)
    def _filter_orders(res_obj: Any) -> list[tuple[Any, ...]]:
        ords = res_obj.orders
        if ords is None or ords.empty:
            return []
        ot = pd.to_datetime(ords["time"])
        sub = ords[ot <= t_time] if is_full_length_perturbed else ords[ot < t_time]
        cols = [c for c in ("order_id", "time", "symbol", "side", "order_type", "quantity", "origin") if c in sub.columns]
        return [tuple(r) for r in sub[cols].itertuples(index=False, name=None)]

    if _filter_orders(full_res) != _filter_orders(prefix_res):
        violations.append(
            LookaheadViolation(
                bar_index=t_bar,
                time=t_time,
                check_type=check_kind,
                source="ENGINE",
                details=f"Danh sách Orders tạo tới bar {t_bar} khác nhau ({check_kind}).",
            )
        )

    # 4. Fills khớp tại bar_index <= t_bar
    ev_f = full_res.order_events
    ev_p = prefix_res.order_events
    fil_f = ev_f[(ev_f["event_type"] == "FILLED") & (ev_f["bar_index"] <= t_bar)] if not ev_f.empty else ev_f
    fil_p = ev_p[(ev_p["event_type"] == "FILLED") & (ev_p["bar_index"] <= t_bar)] if not ev_p.empty else ev_p
    if len(fil_f) != len(fil_p):
        violations.append(
            LookaheadViolation(
                bar_index=t_bar,
                time=t_time,
                check_type=check_kind,
                source="ENGINE",
                details=f"Số lượng Fills đến bar {t_bar} khác nhau ({len(fil_f)} vs {len(fil_p)}).",
            )
        )

    # 5. Ledger entries đến t_time
    led_f = full_res.ledger[pd.to_datetime(full_res.ledger["time"]) <= t_time] if not full_res.ledger.empty else full_res.ledger
    led_p = prefix_res.ledger[pd.to_datetime(prefix_res.ledger["time"]) <= t_time] if not prefix_res.ledger.empty else prefix_res.ledger
    if len(led_f) != len(led_p):
        violations.append(
            LookaheadViolation(
                bar_index=t_bar,
                time=t_time,
                check_type=check_kind,
                source="ENGINE",
                details=f"Số lượng Ledger entries đến bar {t_bar} khác nhau ({len(led_f)} vs {len(led_p)}).",
            )
        )
    return violations


def verify_no_lookahead(
    market_data: pd.DataFrame,
    strategy: Any,
    config: EngineConfig | None = None,
    validation_config: ValidationConfig | None = None,
    *,
    sample_indices: Sequence[int] | None = None,
    check_all_bars: bool | None = None,
    sample_step: int | None = None,
    check_backtest_state: bool | None = None,
    run_mode: StrategyMode | None = None,
) -> LookaheadAuditReport:
    """
    Kiểm định toàn diện Anti-Lookahead Bias cho Strategy và Engine (H1).
    Không thay đổi market_data, config hay strategy instance gốc.
    """
    if market_data is None or market_data.empty or len(market_data) < 3:
        return LookaheadAuditReport(
            status=ValidationStatus.NOT_RUN,
            strategy_status=ValidationStatus.NOT_RUN,
            engine_status=ValidationStatus.NOT_RUN,
            messages=["INSUFFICIENT_DATA: Cần ít nhất 3 nến để kiểm tra Anti-Lookahead."],
        )

    vcfg = validation_config if validation_config is not None else ValidationConfig()
    eff_check_all = vcfg.check_all_bars if check_all_bars is None else bool(check_all_bars)
    eff_step = vcfg.sample_step if sample_step is None else int(sample_step)
    eff_check_state = vcfg.check_backtest_state if check_backtest_state is None else bool(check_backtest_state)

    df = market_data.copy().reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    cfg_no_fc = clone_config(config, force_close_at_end=False)
    symbol = cfg_no_fc.instrument.symbol

    probe_strat = instantiate_fresh_strategy(strategy)
    warmup = int(getattr(probe_strat, "warmup_period", 0) or 0)
    strat_mode = run_mode if run_mode is not None else getattr(probe_strat, "mode", StrategyMode.VECTORIZED)
    if strat_mode == StrategyMode.BOTH:
        strat_mode = StrategyMode.VECTORIZED

    checked_bars = _resolve_sample_bars(
        len(df),
        warmup,
        check_all_bars=eff_check_all,
        sample_indices=sample_indices,
        sample_step=eff_step,
        max_samples=None if eff_check_all else vcfg.max_lookahead_samples,
        random_seed=vcfg.random_seed,
    )
    if not checked_bars:
        return LookaheadAuditReport(
            status=ValidationStatus.NOT_RUN,
            strategy_status=ValidationStatus.NOT_RUN,
            engine_status=ValidationStatus.NOT_RUN,
            messages=["INSUFFICIENT_POST_WARMUP_BARS: Không đủ nến sau warm-up để kiểm tra Lookahead."],
        )

    violations: list[LookaheadViolation] = []
    messages: list[str] = []
    validator = DataValidator()

    # 1. Chạy Full Backtest (force_close_at_end=False) để kiểm tra Order Eligibility & đối chiếu Prefix
    full_res = BacktestEngine(cfg_no_fc).run(
        df.copy(), instantiate_fresh_strategy(strategy), run_mode=strat_mode
    )
    violations.extend(audit_order_eligibility(full_res))

    # 2. Kiểm tra Causal Signal Invariance & Future Perturbation (VECTORIZED)
    full_vec_signals: list[Any] | None = None
    if strat_mode == StrategyMode.VECTORIZED:
        full_vec_signals = _extract_vectorized_signals(df, strategy, symbol)

    pert_factor = float(vcfg.perturbation_factors[0])
    for t in checked_bars:
        t_time = pd.Timestamp(times.iloc[t])
        prefix_df = df.iloc[: t + 1].copy().reset_index(drop=True)

        # Future Perturbation + DataValidator guard
        pert_df = _perturb_future_bars(df, split_bar=t, factor=pert_factor)
        val_report = validator.validate(pert_df)
        if not val_report.is_valid:
            messages.append(
                f"PERTURBATION_VALIDATION_WARNING at bar {t}: {val_report.errors}"
            )

        if strat_mode == StrategyMode.VECTORIZED and full_vec_signals is not None:
            pref_sigs = _extract_vectorized_signals(prefix_df, strategy, symbol)
            if not _signals_equal(full_vec_signals[t], pref_sigs[t]):
                violations.append(
                    LookaheadViolation(
                        bar_index=t,
                        time=t_time,
                        check_type="PREFIX_SIGNAL",
                        source="STRATEGY",
                        details=(
                            f"Tín hiệu tại bar {t} thay đổi khi cắt ngắn tương lai: "
                            f"full={full_vec_signals[t].signal_type.name} vs prefix={pref_sigs[t].signal_type.name}."
                        ),
                    )
                )
            pert_sigs = _extract_vectorized_signals(pert_df, strategy, symbol)
            for k in range(warmup, t + 1):
                if not _signals_equal(full_vec_signals[k], pert_sigs[k]):
                    violations.append(
                        LookaheadViolation(
                            bar_index=k,
                            time=pd.Timestamp(times.iloc[k]),
                            check_type="FUTURE_PERTURBATION",
                            source="STRATEGY",
                            details=(
                                f"Tín hiệu tại bar {k} thay đổi khi gây nhiễu tương lai sau bar {t}."
                            ),
                        )
                    )
                    break

        # Prefix Backtest State Equivalence & Future Perturbation ở tầng Engine (kể cả EVENT_DRIVEN)
        if eff_check_state:
            prefix_res = BacktestEngine(cfg_no_fc).run(
                prefix_df, instantiate_fresh_strategy(strategy), run_mode=strat_mode
            )
            violations.extend(
                _compare_prefix_backtest_state(
                    full_res, prefix_res, t, t_time, is_full_length_perturbed=False
                )
            )

            if strat_mode == StrategyMode.EVENT_DRIVEN:
                pert_res = BacktestEngine(cfg_no_fc).run(
                    pert_df, instantiate_fresh_strategy(strategy), run_mode=strat_mode
                )
                violations.extend(
                    _compare_prefix_backtest_state(
                        full_res, pert_res, t, t_time, is_full_length_perturbed=True
                    )
                )

    strat_v = [v for v in violations if v.source == "STRATEGY"]
    eng_v = [v for v in violations if v.source == "ENGINE"]
    strat_status = ValidationStatus.FAIL if strat_v else ValidationStatus.PASS
    eng_status = ValidationStatus.FAIL if eng_v else ValidationStatus.PASS
    overall = worst_status(strat_status, eng_status)
    return LookaheadAuditReport(
        status=overall,
        strategy_status=strat_status,
        engine_status=eng_status,
        checked_bars=checked_bars,
        violations=violations,
        messages=messages,
    )
