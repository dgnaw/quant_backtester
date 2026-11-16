"""Package Validation & Backtest Integrity (Group H – Gate 04)."""
from __future__ import annotations

from .config import (
    MarketRegimeConfig,
    OverfittingThresholds,
    StabilityConfig,
    StressTestConfig,
    ValidationConfig,
)
from .grid_runner import (
    GridRunResult,
    build_param_grid,
    grid_runs_to_dataframe,
    instantiate_fresh_strategy,
    run_backtest,
    run_parameter_grid,
    select_best_run,
)
from .integrity_report import (
    DEFAULT_MANDATORY_CHECKS,
    BacktestIntegrityReport,
    assemble_integrity_report,
    run_integrity_suite,
)
from .lookahead import (
    LookaheadAuditReport,
    LookaheadViolation,
    audit_order_eligibility,
    verify_no_lookahead,
)
from .models import ValidationError, ValidationStatus, worst_status
from .overfitting import (
    OverfittingReport,
    adjust_multiple_testing_pvalues,
    calculate_deflated_sharpe_ratio,
    calculate_probabilistic_sharpe_ratio,
    calculate_spearman_rank_correlation,
    evaluate_overfitting,
    expected_maximum_sharpe_ratio,
)
from .parameter_stability import (
    ParameterStabilityReport,
    analyze_parameter_stability,
)
from .regime_analysis import (
    MarketRegimeReport,
    analyze_market_regimes,
    classify_market_regimes,
    extract_contiguous_segments,
)
from .reproducibility import (
    ReproducibilityReport,
    classify_fingerprint_pair,
    compute_input_fingerprint,
    compute_result_fingerprint,
    deterministic_seed_context,
    make_rng,
    verify_reproducibility,
)
from .stress_test import (
    FrictionStressReport,
    StressScenarioResult,
    estimate_break_even_multiplier,
    run_controlled_commission_test,
    run_friction_stress_test,
)
from .train_test import (
    ForceCloseSummary,
    TrainTestEvaluationResult,
    TrainTestSplit,
    run_in_sample_out_of_sample,
    split_in_sample_out_of_sample,
)
from .walk_forward import (
    WalkForwardFoldResult,
    WalkForwardMode,
    WalkForwardResult,
    run_walk_forward,
)
from .warmup_guard import (
    EvaluationWindow,
    WarmupAuditReport,
    compute_evaluation_metrics,
    extract_evaluation_window,
    resolve_boundaries,
    verify_warmup_boundary,
)

__all__ = [
    "BacktestIntegrityReport",
    "DEFAULT_MANDATORY_CHECKS",
    "EvaluationWindow",
    "ForceCloseSummary",
    "FrictionStressReport",
    "GridRunResult",
    "LookaheadAuditReport",
    "LookaheadViolation",
    "MarketRegimeConfig",
    "MarketRegimeReport",
    "OverfittingReport",
    "OverfittingThresholds",
    "ParameterStabilityReport",
    "ReproducibilityReport",
    "StabilityConfig",
    "StressScenarioResult",
    "StressTestConfig",
    "TrainTestEvaluationResult",
    "TrainTestSplit",
    "ValidationConfig",
    "ValidationError",
    "ValidationStatus",
    "WalkForwardFoldResult",
    "WalkForwardMode",
    "WalkForwardResult",
    "WarmupAuditReport",
    "adjust_multiple_testing_pvalues",
    "analyze_market_regimes",
    "analyze_parameter_stability",
    "assemble_integrity_report",
    "audit_order_eligibility",
    "build_param_grid",
    "calculate_deflated_sharpe_ratio",
    "calculate_probabilistic_sharpe_ratio",
    "calculate_spearman_rank_correlation",
    "classify_fingerprint_pair",
    "classify_market_regimes",
    "compute_evaluation_metrics",
    "compute_input_fingerprint",
    "compute_result_fingerprint",
    "deterministic_seed_context",
    "estimate_break_even_multiplier",
    "evaluate_overfitting",
    "expected_maximum_sharpe_ratio",
    "extract_contiguous_segments",
    "extract_evaluation_window",
    "grid_runs_to_dataframe",
    "instantiate_fresh_strategy",
    "make_rng",
    "resolve_boundaries",
    "run_backtest",
    "run_controlled_commission_test",
    "run_friction_stress_test",
    "run_in_sample_out_of_sample",
    "run_integrity_suite",
    "run_parameter_grid",
    "run_walk_forward",
    "select_best_run",
    "split_in_sample_out_of_sample",
    "verify_no_lookahead",
    "verify_reproducibility",
    "verify_warmup_boundary",
    "worst_status",
]
