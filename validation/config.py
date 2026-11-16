"""Cấu hình cho Group H – Validation & Backtest Integrity (tách biệt khỏi EngineConfig)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math


def _check_positive_int(name: str, value: int, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} phải là số nguyên.")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} phải {'>= 0' if allow_zero else '> 0'} (nhận {value}).")


@dataclass
class ValidationConfig:
    """Cấu hình chung cho các kiểm định Group H. Đây là nguồn seed DUY NHẤT của Validation."""

    random_seed: int = 42
    deterministic: bool = True

    # H1 – Anti-Lookahead sampling
    check_all_bars: bool = False
    sample_step: int = 10
    max_lookahead_samples: int | None = 20
    check_backtest_state: bool = True
    perturbation_factors: tuple[float, ...] = (1.25, 0.8)

    # H2 – Warm-up
    strict_signal_start: bool = False

    # H3 – IS/OOS
    train_ratio: float = 0.7
    embargo_bars: int = 0

    # H4 – Walk-Forward
    walk_forward_folds: int = 5
    min_train_trades: int = 1
    objective_metric: str = "sharpe_annual"

    # H5 – Reproducibility
    fingerprint_precision: int = 10
    reproducibility_runs: int = 2

    # Gate 04
    strict_gate: bool = True

    def __post_init__(self):
        _check_positive_int("random_seed", self.random_seed, allow_zero=True)
        _check_positive_int("sample_step", self.sample_step)
        if self.max_lookahead_samples is not None:
            _check_positive_int("max_lookahead_samples", self.max_lookahead_samples)
        if not math.isfinite(float(self.train_ratio)) or not (0.0 < float(self.train_ratio) < 1.0):
            raise ValueError("train_ratio phải nằm trong khoảng (0, 1).")
        _check_positive_int("embargo_bars", self.embargo_bars, allow_zero=True)
        _check_positive_int("walk_forward_folds", self.walk_forward_folds)
        if self.walk_forward_folds < 2:
            raise ValueError("walk_forward_folds phải >= 2.")
        _check_positive_int("min_train_trades", self.min_train_trades, allow_zero=True)
        _check_positive_int("fingerprint_precision", self.fingerprint_precision)
        _check_positive_int("reproducibility_runs", self.reproducibility_runs)
        if self.reproducibility_runs < 2:
            raise ValueError("reproducibility_runs phải >= 2.")
        factors = tuple(float(f) for f in self.perturbation_factors)
        if not factors or any((not math.isfinite(f)) or f <= 0.0 or f == 1.0 for f in factors):
            raise ValueError("perturbation_factors phải là các số dương hữu hạn khác 1.0.")
        self.perturbation_factors = factors


@dataclass
class StabilityConfig:
    """Ngưỡng cấu hình cho phát hiện Isolated Peak (H7). Không hard-code trong thuật toán."""

    max_relative_gap: float = 0.5
    neighborhood_floor: float = 0.0
    max_neighbor_dispersion: float = 0.5
    min_profitable_ratio: float = 0.5
    min_neighbors: int = 1


@dataclass
class MarketRegimeConfig:
    """Cấu hình phân loại Regime (H8). Mọi ngưỡng đều tính nhân quả."""

    trend_window: int = 20
    vol_window: int = 20
    trend_threshold: float = 1.0
    vol_threshold_method: str = "EXPANDING_MEDIAN"
    vol_threshold_window: int = 100
    vol_percentile: float = 0.5
    fixed_vol_threshold: float | None = None
    min_vol_history: int | None = None

    def __post_init__(self):
        _check_positive_int("trend_window", self.trend_window)
        _check_positive_int("vol_window", self.vol_window)
        if self.trend_window < 2 or self.vol_window < 2:
            raise ValueError("trend_window và vol_window phải >= 2.")
        if not math.isfinite(float(self.trend_threshold)) or float(self.trend_threshold) < 0.0:
            raise ValueError("trend_threshold phải >= 0.")
        method = str(self.vol_threshold_method).upper()
        if method not in {"EXPANDING_MEDIAN", "ROLLING_MEDIAN", "ROLLING_PERCENTILE", "FIXED"}:
            raise ValueError(f"vol_threshold_method '{self.vol_threshold_method}' không được hỗ trợ.")
        self.vol_threshold_method = method
        if method == "FIXED" and (self.fixed_vol_threshold is None or float(self.fixed_vol_threshold) < 0.0):
            raise ValueError("FIXED yêu cầu fixed_vol_threshold >= 0.")
        if not (0.0 < float(self.vol_percentile) < 1.0):
            raise ValueError("vol_percentile phải nằm trong khoảng (0, 1).")
        _check_positive_int("vol_threshold_window", self.vol_threshold_window)


@dataclass
class OverfittingThresholds:
    """Ngưỡng ra quyết định Overfitting (H9). Không kết luận dựa trên một chỉ số duy nhất."""

    min_dsr_threshold: float = 0.95
    min_wfe_threshold: float = 0.5
    min_rank_correlation: float = 0.0
    max_is_oos_degradation: float = 0.5
    fail_flag_count: int = 2
    multiple_testing_method: str = "holm"


DEFAULT_STRESS_MULTIPLIERS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass
class StressTestConfig:
    multipliers: tuple[float, ...] = DEFAULT_STRESS_MULTIPLIERS
    stress_types: tuple[str, ...] = ("SPREAD", "COMMISSION", "SLIPPAGE", "COMBINED")
    extra: dict = field(default_factory=dict)
