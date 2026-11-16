"""H9 – Multiple Testing & Overfitting Control (PSR, DSR, Multiple Testing Correction, IS/OOS Diagnostics)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_backtester.validation.config import OverfittingThresholds
from quant_backtester.validation.models import ValidationStatus

_STD_NORM = NormalDist(0.0, 1.0)
_EULER_GAMMA = 0.5772156649015329


def _normal_cdf(z: float) -> float:
    return _STD_NORM.cdf(float(z))


def _normal_ppf(p: float) -> float:
    p_clamped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return _STD_NORM.inv_cdf(p_clamped)


def calculate_probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    n_observations: int = 252,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float | None:
    """
    Probabilistic Sharpe Ratio (PSR) với observed_sharpe và benchmark_sharpe CHƯA annualize (trên từng bar).
    excess_kurtosis là Fisher's Excess Kurtosis (kurtosis phân phối chuẩn = 0.0).
    Trả về xác suất trong [0.0, 1.0], hoặc None nếu dữ liệu không đủ / phương sai ước lượng không hợp lệ.
    """
    if (
        isinstance(n_observations, bool)
        or not isinstance(n_observations, int)
        or n_observations < 3
    ):
        return None
    vals = (observed_sharpe, benchmark_sharpe, skewness, excess_kurtosis)
    if any(isinstance(v, bool) or not math.isfinite(float(v)) for v in vals):
        return None

    sr = float(observed_sharpe)
    sr_b = float(benchmark_sharpe)
    g3 = float(skewness)
    g4 = float(excess_kurtosis)

    numerator = 1.0 - g3 * sr + ((g4 + 2.0) / 4.0) * (sr ** 2)
    variance_sr = numerator / float(n_observations - 1)
    if variance_sr <= 0.0 or not math.isfinite(variance_sr):
        return None

    sigma_sr = math.sqrt(variance_sr)
    z = (sr - sr_b) / sigma_sr
    return float(_normal_cdf(z))


def expected_maximum_sharpe_ratio(
    n_trials: int,
    trials_sharpe_std: float,
    euler_gamma: float = _EULER_GAMMA,
) -> float | None:
    """
    Tính ngưỡng Expected Maximum Sharpe Ratio (SR_0) dưới giả thuyết vô hiệu khi thử N lần:
        SR_0 = sigma_trials * [ (1 - gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e)) ]
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        return None
    if isinstance(trials_sharpe_std, bool) or not math.isfinite(float(trials_sharpe_std)) or float(trials_sharpe_std) < 0.0:
        return None
    if n_trials == 1:
        return 0.0
    z1 = _normal_ppf(1.0 - 1.0 / float(n_trials))
    z2 = _normal_ppf(1.0 - 1.0 / (float(n_trials) * math.e))
    return float(float(trials_sharpe_std) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2))


def calculate_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    trials_sharpe_std: float,
    n_observations: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> tuple[float | None, float | None]:
    """
    Deflated Sharpe Ratio (DSR = PSR(SR_obs, SR_0)).
    Trả về cặp (dsr, expected_max_sharpe_null).
    """
    sr0 = expected_maximum_sharpe_ratio(n_trials, trials_sharpe_std)
    if sr0 is None:
        return None, None
    dsr = calculate_probabilistic_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=sr0,
        n_observations=n_observations,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
    return dsr, sr0


def adjust_multiple_testing_pvalues(
    p_values: Sequence[float],
    method: str = "holm",
) -> np.ndarray:
    """
    Hiệu chỉnh p-value cho đa kiểm định ('bonferroni', 'sidak', 'holm').
    Giữ nguyên thứ tự phần tử ban đầu.
    """
    arr = np.asarray(p_values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("p_values phải là vector 1 chiều khác rỗng.")
    if np.any(~np.isfinite(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("Mọi p-value phải hữu hạn và nằm trong [0.0, 1.0].")

    m = len(arr)
    meth = str(method).strip().lower()
    if meth == "bonferroni":
        return np.minimum(1.0, arr * float(m))
    if meth == "sidak":
        return np.clip(1.0 - np.power(1.0 - arr, float(m)), 0.0, 1.0)
    if meth == "holm":
        order = np.argsort(arr, kind="mergesort")
        sorted_p = arr[order]
        adj_sorted = np.empty(m, dtype=float)
        running_max = 0.0
        for idx, p_i in enumerate(sorted_p):
            candidate = min(1.0, float(m - idx) * float(p_i))
            running_max = max(running_max, candidate)
            adj_sorted[idx] = running_max
        out = np.empty(m, dtype=float)
        out[order] = adj_sorted
        return out
    raise ValueError(f"Phương pháp '{method}' không được hỗ trợ. Hỗ trợ: 'bonferroni', 'sidak', 'holm'.")


def calculate_spearman_rank_correlation(
    is_scores: Sequence[float | None],
    oos_scores: Sequence[float | None],
) -> float | None:
    """
    Tính tương quan hạng Spearman giữa IS_Scores và OOS_Scores:
    - Dùng Average Tie Ranking (pd.Series.rank(method='average')).
    - Trả về None nếu < 3 cặp hợp lệ hoặc toàn bộ hạng trùng nhau (std == 0).
    """
    if len(is_scores) != len(oos_scores):
        raise ValueError("is_scores và oos_scores phải có cùng độ dài.")
    pairs = []
    for a, b in zip(is_scores, oos_scores):
        if a is None or b is None:
            continue
        fa, fb = float(a), float(b)
        if math.isfinite(fa) and math.isfinite(fb):
            pairs.append((fa, fb))
    if len(pairs) < 3:
        return None
    s_is = pd.Series([p[0] for p in pairs]).rank(method="average").to_numpy(dtype=float)
    s_oos = pd.Series([p[1] for p in pairs]).rank(method="average").to_numpy(dtype=float)
    std_is = float(np.std(s_is, ddof=1))
    std_oos = float(np.std(s_oos, ddof=1))
    if std_is <= 1e-12 or std_oos <= 1e-12:
        return None
    corr = float(np.corrcoef(s_is, s_oos)[0, 1])
    return max(-1.0, min(1.0, corr)) if math.isfinite(corr) else None


@dataclass
class OverfittingReport:
    status: ValidationStatus
    probabilistic_sharpe_ratio: float | None
    deflated_sharpe_ratio: float | None
    expected_max_sharpe_null: float | None
    n_trials: int
    raw_p_value: float | None
    adjusted_p_value: float | None
    is_oos_return_degradation: float | None
    is_oos_sharpe_degradation: float | None
    walk_forward_efficiency: float | None
    is_oos_rank_correlation: float | None
    triggered_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assumptions_and_limitations: list[str] = field(default_factory=list)


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    fn, fd = float(num), float(den)
    if not (math.isfinite(fn) and math.isfinite(fd)) or abs(fd) <= 1e-12:
        return None
    return fn / fd


def evaluate_overfitting(
    *,
    observed_sharpe_annual: float | None,
    bars_per_year: int = 96 * 252,
    n_observations: int = 0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    trial_sharpes_annual: Sequence[float] | None = None,
    n_trials: int | None = None,
    is_annualized_return_pct: float | None = None,
    oos_annualized_return_pct: float | None = None,
    is_sharpe_annual: float | None = None,
    oos_sharpe_annual: float | None = None,
    walk_forward_efficiency: float | None = None,
    is_grid_scores: Sequence[float | None] | None = None,
    oos_grid_scores: Sequence[float | None] | None = None,
    thresholds: OverfittingThresholds | None = None,
) -> OverfittingReport:
    """
    Tổng hợp đánh giá rủi ro Overfitting & Multiple Testing (H9).
    Quy đổi Sharpe annual -> Sharpe per-bar trước khi đưa vào PSR/DSR.
    Không kết luận Overfit chỉ dựa trên một chỉ số đơn lẻ.
    """
    cfg = thresholds if thresholds is not None else OverfittingThresholds()
    assumptions = [
        "PSR/DSR giả định cấu trúc i.i.d. xấp xỉ cho sai số ước lượng của Sharpe; lợi suất tài chính có thể có tự tương quan.",
        "V1.0 dùng số lượng trials thực tế n_trials như xấp xỉ; các trials trên lưới tham số gần nhau có tương quan nên số trials độc lập hiệu dụng có thể nhỏ hơn.",
        "PSR, DSR, WFE và Spearman Rank Correlation là thước đo định lượng hỗ trợ cảnh báo rủi ro, không khẳng định chắc chắn lợi nhuận tương lai.",
    ]

    sqrt_bpy = math.sqrt(float(bars_per_year)) if bars_per_year > 0 else 1.0
    sr_bar = (
        float(observed_sharpe_annual) / sqrt_bpy
        if observed_sharpe_annual is not None and math.isfinite(float(observed_sharpe_annual))
        else None
    )

    clean_trials = (
        [float(s) / sqrt_bpy for s in trial_sharpes_annual if s is not None and math.isfinite(float(s))]
        if trial_sharpes_annual is not None
        else []
    )
    eff_trials = int(n_trials) if n_trials is not None else max(1, len(clean_trials))
    trials_std_bar = float(np.std(clean_trials, ddof=1)) if len(clean_trials) >= 2 else 0.0

    psr: float | None = None
    dsr: float | None = None
    sr0_bar: float | None = None
    sr0_annual: float | None = None
    raw_p: float | None = None
    adj_p: float | None = None

    if sr_bar is not None and n_observations >= 3:
        psr = calculate_probabilistic_sharpe_ratio(
            observed_sharpe=sr_bar,
            benchmark_sharpe=0.0,
            n_observations=n_observations,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
        )
        dsr, sr0_bar = calculate_deflated_sharpe_ratio(
            observed_sharpe=sr_bar,
            n_trials=eff_trials,
            trials_sharpe_std=trials_std_bar,
            n_observations=n_observations,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
        )
        if sr0_bar is not None:
            sr0_annual = sr0_bar * sqrt_bpy
        if psr is not None:
            raw_p = 1.0 - psr
            if len(clean_trials) >= 1:
                trial_p = []
                for t_sr in clean_trials:
                    p_t = calculate_probabilistic_sharpe_ratio(
                        t_sr, 0.0, n_observations, skewness, excess_kurtosis
                    )
                    trial_p.append(1.0 - p_t if p_t is not None else 1.0)
                adj_all = adjust_multiple_testing_pvalues(trial_p, method=cfg.multiple_testing_method)
                # Nếu observed trùng một trial, lấy p hiệu chỉnh tương ứng hoặc min với công thức chuẩn
                adj_p = float(adjust_multiple_testing_pvalues([raw_p] + [1.0] * max(0, eff_trials - 1), method=cfg.multiple_testing_method)[0]) if eff_trials > len(clean_trials) else float(np.min(adj_all))
            else:
                adj_p = float(adjust_multiple_testing_pvalues([raw_p] + [1.0] * max(0, eff_trials - 1), method=cfg.multiple_testing_method)[0])

    ret_deg = _safe_div(oos_annualized_return_pct, is_annualized_return_pct)
    sr_deg = _safe_div(oos_sharpe_annual, is_sharpe_annual)
    rank_corr = (
        calculate_spearman_rank_correlation(is_grid_scores, oos_grid_scores)
        if is_grid_scores is not None and oos_grid_scores is not None
        else None
    )

    flags: list[str] = []
    warnings: list[str] = []
    if dsr is not None and dsr < cfg.min_dsr_threshold:
        flags.append("LOW_DSR")
        warnings.append(f"DSR={dsr:.4f} < ngưỡng {cfg.min_dsr_threshold:.2f} (rủi ro Multiple Testing).")
    if walk_forward_efficiency is not None and walk_forward_efficiency < cfg.min_wfe_threshold:
        flags.append("LOW_WFE")
        warnings.append(f"WFE={walk_forward_efficiency:.4f} < ngưỡng {cfg.min_wfe_threshold:.2f}.")
    if rank_corr is not None and rank_corr < cfg.min_rank_correlation:
        flags.append("LOW_RANK_CORRELATION")
        warnings.append(f"Spearman Rank Correlation IS/OOS={rank_corr:.4f} < {cfg.min_rank_correlation:.2f}.")
    if sr_deg is not None and sr_deg < cfg.max_is_oos_degradation:
        flags.append("HIGH_SHARPE_DEGRADATION")
        warnings.append(f"IS->OOS Sharpe Retention={sr_deg:.4f} < {cfg.max_is_oos_degradation:.2f}.")

    if len(flags) >= cfg.fail_flag_count:
        status = ValidationStatus.FAIL
    elif flags:
        status = ValidationStatus.WARNING
    elif psr is None and walk_forward_efficiency is None and rank_corr is None:
        status = ValidationStatus.NOT_RUN
    else:
        status = ValidationStatus.PASS

    return OverfittingReport(
        status=status,
        probabilistic_sharpe_ratio=psr,
        deflated_sharpe_ratio=dsr,
        expected_max_sharpe_null=sr0_annual,
        n_trials=eff_trials,
        raw_p_value=raw_p,
        adjusted_p_value=adj_p,
        is_oos_return_degradation=ret_deg,
        is_oos_sharpe_degradation=sr_deg,
        walk_forward_efficiency=walk_forward_efficiency,
        is_oos_rank_correlation=rank_corr,
        triggered_flags=flags,
        warnings=warnings,
        assumptions_and_limitations=assumptions,
    )
