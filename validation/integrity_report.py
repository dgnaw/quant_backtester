"""Gate 04 – BacktestIntegrityReport & Validation Suite Orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_backtester.config import EngineConfig
from quant_backtester.models.result import BacktestResult
from quant_backtester.strategies.base import StrategyMode
from quant_backtester.validation.config import (
    MarketRegimeConfig,
    OverfittingThresholds,
    StabilityConfig,
    ValidationConfig,
)
from quant_backtester.validation.grid_runner import (
    clone_config,
    instantiate_fresh_strategy,
    run_backtest,
    run_parameter_grid,
)
from quant_backtester.validation.lookahead import LookaheadAuditReport, verify_no_lookahead
from quant_backtester.validation.models import ValidationStatus, worst_status
from quant_backtester.validation.overfitting import OverfittingReport, evaluate_overfitting
from quant_backtester.validation.parameter_stability import (
    ParameterStabilityReport,
    analyze_parameter_stability,
)
from quant_backtester.validation.regime_analysis import (
    MarketRegimeReport,
    analyze_market_regimes,
)
from quant_backtester.validation.reproducibility import (
    ReproducibilityReport,
    verify_reproducibility,
)
from quant_backtester.validation.stress_test import (
    FrictionStressReport,
    run_friction_stress_test,
)
from quant_backtester.validation.train_test import (
    TrainTestEvaluationResult,
    run_in_sample_out_of_sample,
)
from quant_backtester.validation.walk_forward import (
    WalkForwardMode,
    WalkForwardResult,
    run_walk_forward,
)
from quant_backtester.validation.warmup_guard import WarmupAuditReport, verify_warmup_boundary

DEFAULT_MANDATORY_CHECKS: tuple[str, ...] = (
    "H1_LOOKAHEAD",
    "H2_WARMUP",
    "H5_REPRODUCIBILITY",
)


@dataclass
class BacktestIntegrityReport:
    overall_status: ValidationStatus
    gate_04_passed: bool
    strict_gate: bool
    mandatory_checks: tuple[str, ...]
    statuses: dict[str, ValidationStatus]
    baseline_result: BacktestResult | None = None
    lookahead_report: LookaheadAuditReport | None = None
    warmup_report: WarmupAuditReport | None = None
    train_test_result: TrainTestEvaluationResult | None = None
    walk_forward_result: WalkForwardResult | None = None
    reproducibility_report: ReproducibilityReport | None = None
    stress_report: FrictionStressReport | None = None
    stability_report: ParameterStabilityReport | None = None
    regime_report: MarketRegimeReport | None = None
    overfitting_report: OverfittingReport | None = None
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "gate_04_passed": self.gate_04_passed,
            "strict_gate": self.strict_gate,
            "statuses": {k: v.value for k, v in self.statuses.items()},
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


def assemble_integrity_report(
    *,
    baseline_result: BacktestResult | None = None,
    lookahead_report: LookaheadAuditReport | None = None,
    warmup_report: WarmupAuditReport | None = None,
    train_test_result: TrainTestEvaluationResult | None = None,
    walk_forward_result: WalkForwardResult | None = None,
    reproducibility_report: ReproducibilityReport | None = None,
    stress_report: FrictionStressReport | None = None,
    stability_report: ParameterStabilityReport | None = None,
    regime_report: MarketRegimeReport | None = None,
    overfitting_report: OverfittingReport | None = None,
    strict_gate: bool = True,
    mandatory_checks: Sequence[str] = DEFAULT_MANDATORY_CHECKS,
) -> BacktestIntegrityReport:
    """Tổng hợp 9 kết quả H1–H9 thành BacktestIntegrityReport và quyết định Gate 04."""
    statuses: dict[str, ValidationStatus] = {
        "H1_LOOKAHEAD": lookahead_report.status if lookahead_report is not None else ValidationStatus.NOT_RUN,
        "H2_WARMUP": warmup_report.status if warmup_report is not None else ValidationStatus.NOT_RUN,
        "H3_IS_OOS": train_test_result.status if train_test_result is not None else ValidationStatus.NOT_RUN,
        "H4_WALK_FORWARD": walk_forward_result.status if walk_forward_result is not None else ValidationStatus.NOT_RUN,
        "H5_REPRODUCIBILITY": reproducibility_report.status if reproducibility_report is not None else ValidationStatus.NOT_RUN,
        "H6_STRESS_TEST": stress_report.status if stress_report is not None else ValidationStatus.NOT_RUN,
        "H7_PARAMETER_STABILITY": stability_report.status if stability_report is not None else ValidationStatus.NOT_RUN,
        "H8_REGIME_ANALYSIS": regime_report.status if regime_report is not None else ValidationStatus.NOT_RUN,
        "H9_OVERFITTING": overfitting_report.status if overfitting_report is not None else ValidationStatus.NOT_RUN,
    }

    warnings: list[str] = []
    failures: list[str] = []
    mandatory = tuple(mandatory_checks)

    for key, st in statuses.items():
        if st == ValidationStatus.FAIL:
            if key in mandatory or strict_gate:
                failures.append(f"{key}: FAIL")
            else:
                warnings.append(f"{key}: FAIL (non-mandatory, strict_gate=False)")
        elif st == ValidationStatus.WARNING:
            warnings.append(f"{key}: WARNING")
        elif st == ValidationStatus.NOT_RUN and key in mandatory and strict_gate:
            failures.append(f"{key}: NOT_RUN (bắt buộc trong strict_gate)")

    # Gom thông điệp chi tiết từ các report con
    if lookahead_report is not None and lookahead_report.violations:
        for v in lookahead_report.violations[:10]:
            failures.append(f"H1[{v.source}/{v.check_type}@{v.bar_index}]: {v.details}")
    if warmup_report is not None and warmup_report.violations:
        for msg in warmup_report.violations[:10]:
            failures.append(f"H2: {msg}")
    if reproducibility_report is not None and reproducibility_report.differences:
        for msg in reproducibility_report.differences[:10]:
            failures.append(f"H5: {msg}")
    if stress_report is not None:
        warnings.extend(f"H6: {m}" for m in stress_report.messages)
    if stability_report is not None:
        warnings.extend(f"H7: {m}" for m in stability_report.warnings)
    if regime_report is not None:
        warnings.extend(f"H8: {m}" for m in regime_report.messages)
    if overfitting_report is not None:
        warnings.extend(f"H9: {m}" for m in overfitting_report.warnings)

    gate_passed = len(failures) == 0
    if not gate_passed:
        overall = ValidationStatus.FAIL
    elif warnings:
        overall = ValidationStatus.WARNING
    elif all(st == ValidationStatus.NOT_RUN for st in statuses.values()):
        overall = ValidationStatus.NOT_RUN
    else:
        overall = worst_status(*statuses.values())
        if overall == ValidationStatus.FAIL and not strict_gate:
            overall = ValidationStatus.WARNING

    return BacktestIntegrityReport(
        overall_status=overall,
        gate_04_passed=gate_passed,
        strict_gate=strict_gate,
        mandatory_checks=mandatory,
        statuses=statuses,
        baseline_result=baseline_result,
        lookahead_report=lookahead_report,
        warmup_report=warmup_report,
        train_test_result=train_test_result,
        walk_forward_result=walk_forward_result,
        reproducibility_report=reproducibility_report,
        stress_report=stress_report,
        stability_report=stability_report,
        regime_report=regime_report,
        overfitting_report=overfitting_report,
        warnings=warnings,
        failures=failures,
    )


def run_integrity_suite(
    market_data: pd.DataFrame,
    strategy: Any,
    config: EngineConfig | None = None,
    validation_config: ValidationConfig | None = None,
    *,
    baseline_result: BacktestResult | None = None,
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    walk_forward_mode: WalkForwardMode = WalkForwardMode.ROLLING,
    stress_multipliers: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    stress_types: Sequence[str] = ("COMBINED",),
    stability_config: StabilityConfig | None = None,
    regime_config: MarketRegimeConfig | None = None,
    overfitting_thresholds: OverfittingThresholds | None = None,
    mandatory_checks: Sequence[str] = DEFAULT_MANDATORY_CHECKS,
    enable_lookahead: bool = True,
    enable_warmup: bool = True,
    enable_train_test: bool = True,
    enable_walk_forward: bool = True,
    enable_reproducibility: bool = True,
    enable_stress_test: bool = True,
    enable_parameter_stability: bool = True,
    enable_regime_analysis: bool = True,
    enable_overfitting: bool = True,
    run_mode: StrategyMode | None = None,
) -> BacktestIntegrityReport:
    """
    Chạy bộ kiểm định Gate 04 đầy đủ (H1–H9) trên dữ liệu, Strategy và EngineConfig.
    Không sửa đổi market_data, config, strategy gốc hay baseline_result truyền vào.
    """
    vcfg = validation_config if validation_config is not None else ValidationConfig()
    base_cfg = clone_config(config)
    df = market_data.copy()

    base_res = (
        baseline_result
        if baseline_result is not None
        else run_backtest(df, instantiate_fresh_strategy(strategy), base_cfg, run_mode=run_mode)
    )

    lookahead_rep = (
        verify_no_lookahead(df, strategy, base_cfg, vcfg, run_mode=run_mode)
        if enable_lookahead
        else None
    )
    warmup_rep = (
        verify_warmup_boundary(
            base_res,
            strategy=instantiate_fresh_strategy(strategy),
            strict_signal_start=vcfg.strict_signal_start,
        )
        if enable_warmup
        else None
    )
    repro_rep = (
        verify_reproducibility(df, strategy, base_cfg, vcfg)
        if enable_reproducibility
        else None
    )

    tt_res = (
        run_in_sample_out_of_sample(
            df,
            strategy,
            base_cfg,
            train_ratio=vcfg.train_ratio,
            embargo_bars=vcfg.embargo_bars,
        )
        if enable_train_test
        else None
    )

    wf_res = (
        run_walk_forward(
            df,
            strategy,
            base_cfg,
            param_grid=param_grid,
            folds=vcfg.walk_forward_folds,
            mode=walk_forward_mode,
            objective_metric=vcfg.objective_metric,
            min_train_trades=vcfg.min_train_trades,
            embargo_bars=vcfg.embargo_bars,
            run_mode=run_mode,
        )
        if enable_walk_forward
        else None
    )

    stress_rep = (
        run_friction_stress_test(
            df,
            strategy,
            base_cfg,
            multipliers=stress_multipliers,
            stress_types=stress_types,
        )
        if enable_stress_test
        else None
    )

    # DH7.6: Parameter Stability PHẢI chạy trên IS (nếu có IS/OOS) chứ không dùng OOS
    stability_data = tt_res.split.in_sample_data if tt_res is not None else df
    stability_rep = (
        analyze_parameter_stability(
            stability_data,
            strategy,
            base_cfg,
            param_grid=param_grid,
            objective_metric=vcfg.objective_metric,
            min_trades=0,
            stability_config=stability_config,
            run_mode=run_mode,
        )
        if enable_parameter_stability
        else None
    )

    regime_rep = (
        analyze_market_regimes(df, base_res, regime_config=regime_config)
        if enable_regime_analysis
        else None
    )

    overfit_rep: OverfittingReport | None = None
    if enable_overfitting:
        trial_sharpes = None
        n_trials = 1
        is_grid_scores = None
        oos_grid_scores = None
        if stability_rep is not None and not stability_rep.grid_results_df.empty:
            valid_rows = stability_rep.grid_results_df[stability_rep.grid_results_df["is_valid"]]
            trial_sharpes = valid_rows["score"].dropna().astype(float).tolist()
            n_trials = max(1, len(trial_sharpes))
            if tt_res is not None and param_grid is not None:
                oos_runs = run_parameter_grid(
                    tt_res.split.out_of_sample_data,
                    strategy,
                    base_cfg,
                    param_grid=param_grid,
                    objective_metric=vcfg.objective_metric,
                    min_trades=0,
                    trade_start=tt_res.split.oos_trade_start,
                    run_mode=run_mode,
                )
                is_grid_scores = stability_rep.grid_results_df["score"].tolist()
                oos_grid_scores = [r.score for r in oos_runs]

        met = base_res.metrics
        n_obs = max(0, len(base_res.equity) - int(getattr(base_res, "effective_trade_start", 0) or 0) - 1)
        if tt_res is not None:
            n_obs = tt_res.oos_window.return_observations
        obs_sr = (
            tt_res.out_of_sample_metrics.get("sharpe_annual")
            if tt_res is not None
            else met.get("sharpe_annual")
        )
        skew = (
            tt_res.out_of_sample_metrics.get("return_skewness")
            if tt_res is not None
            else met.get("return_skewness")
        )
        kurt = (
            tt_res.out_of_sample_metrics.get("return_kurtosis")
            if tt_res is not None
            else met.get("return_kurtosis")
        )

        overfit_rep = evaluate_overfitting(
            observed_sharpe_annual=obs_sr,
            bars_per_year=int(met.get("bars_per_year", base_cfg.bars_per_year)),
            n_observations=n_obs,
            skewness=float(skew) if skew is not None else 0.0,
            excess_kurtosis=float(kurt) if kurt is not None else 0.0,
            trial_sharpes_annual=trial_sharpes,
            n_trials=n_trials,
            is_annualized_return_pct=wf_res.is_annualized_return_pct if wf_res is not None else (tt_res.in_sample_metrics.get("cagr_pct") if tt_res is not None else None),
            oos_annualized_return_pct=wf_res.oos_annualized_return_pct if wf_res is not None else (tt_res.out_of_sample_metrics.get("cagr_pct") if tt_res is not None else None),
            is_sharpe_annual=tt_res.in_sample_metrics.get("sharpe_annual") if tt_res is not None else None,
            oos_sharpe_annual=tt_res.out_of_sample_metrics.get("sharpe_annual") if tt_res is not None else None,
            walk_forward_efficiency=wf_res.walk_forward_efficiency if wf_res is not None else None,
            is_grid_scores=is_grid_scores,
            oos_grid_scores=oos_grid_scores,
            thresholds=overfitting_thresholds,
        )

    return assemble_integrity_report(
        baseline_result=base_res,
        lookahead_report=lookahead_rep,
        warmup_report=warmup_rep,
        train_test_result=tt_res,
        walk_forward_result=wf_res,
        reproducibility_report=repro_rep,
        stress_report=stress_rep,
        stability_report=stability_rep,
        regime_report=regime_rep,
        overfitting_report=overfit_rep,
        strict_gate=vcfg.strict_gate,
        mandatory_checks=mandatory_checks,
    )
