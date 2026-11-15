from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


def pd_valid(value: Any) -> bool:
    return (
        value is not None
        and not isinstance(value, bool)
        and isinstance(value, (int, float, np.number))
        and bool(np.isfinite(value))
    )


def _clean_returns(returns: pd.Series | np.ndarray | list[float] | None) -> np.ndarray:
    if returns is None:
        return np.array([], dtype=float)
    s = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return s.to_numpy(dtype=float)


def annual_to_bar_rate(risk_free_rate_annual: float, bars_per_year: int) -> float:
    """
    Chuyển đổi lãi suất phi rủi ro năm sang lãi suất ghép lãi mỗi Bar:
        rf_bar = (1 + rf_annual) ** (1 / bars_per_year) - 1
    """
    if isinstance(bars_per_year, bool) or not isinstance(bars_per_year, (int, float)) or bars_per_year <= 0:
        raise ValueError("bars_per_year phải lớn hơn 0.")
    if (
        isinstance(risk_free_rate_annual, bool)
        or not isinstance(risk_free_rate_annual, (int, float))
        or not math.isfinite(float(risk_free_rate_annual))
        or float(risk_free_rate_annual) <= -1.0
    ):
        raise ValueError("risk_free_rate_annual phải là số thực hữu hạn và > -1.0.")
    rf = float(risk_free_rate_annual)
    if abs(rf) <= 1e-15:
        return 0.0
    return float((1.0 + rf) ** (1.0 / float(bars_per_year)) - 1.0)


def calculate_cagr_pct(
    wealth_or_growth: pd.Series | np.ndarray | list[float] | float,
    n_return_bars: int | None = None,
    bars_per_year: int = 96 * 252,
) -> float | None:
    """
    Tính tỷ lệ tăng trưởng kép hàng năm (CAGR %) từ chuỗi Wealth Index hoặc hệ số tăng trưởng:
        cagr_pct = ((growth_factor ** (1 / elapsed_years)) - 1) * 100
    Trả về None nếu không đủ thời gian quan sát hoặc growth_factor <= 0 (không xác định theo lũy thừa).
    """
    if isinstance(bars_per_year, bool) or not isinstance(bars_per_year, (int, float)) or bars_per_year <= 0:
        raise ValueError("bars_per_year phải lớn hơn 0.")

    if isinstance(wealth_or_growth, (pd.Series, np.ndarray, list, tuple)):
        arr = _clean_returns(wealth_or_growth)
        if len(arr) < 2 and (n_return_bars is None or n_return_bars <= 0):
            return None
        if len(arr) == 0:
            return None
        w_init = float(arr[0])
        w_final = float(arr[-1])
        if w_init <= 0.0:
            raise ValueError("Giá trị khởi tạo để tính CAGR phải > 0.")
        growth_factor = w_final / w_init
        eff_bars = int(n_return_bars) if n_return_bars is not None else (len(arr) - 1)
    else:
        if isinstance(wealth_or_growth, bool) or not isinstance(wealth_or_growth, (int, float)):
            raise ValueError("wealth_or_growth phải là chuỗi hoặc số thực.")
        growth_factor = float(wealth_or_growth)
        if not math.isfinite(growth_factor):
            return None
        eff_bars = int(n_return_bars) if n_return_bars is not None else 0

    if eff_bars <= 0:
        return None
    if growth_factor <= 0.0:
        return None

    elapsed_years = float(eff_bars) / float(bars_per_year)
    if elapsed_years <= 0.0:
        return None

    try:
        cagr = (growth_factor ** (1.0 / elapsed_years)) - 1.0
        cagr_pct = float(cagr * 100.0)
    except OverflowError:
        return None
    if not math.isfinite(cagr_pct):
        return None
    return cagr_pct


def calculate_annualized_volatility(
    returns: pd.Series | np.ndarray | list[float] | None,
    bars_per_year: int,
) -> float | None:
    """
    Tính độ lệch chuẩn lợi suất thường niên hóa:
        volatility_annual = std(returns, ddof=1) * sqrt(bars_per_year)
    """
    if isinstance(bars_per_year, bool) or not isinstance(bars_per_year, (int, float)) or bars_per_year <= 0:
        raise ValueError("bars_per_year phải lớn hơn 0.")
    arr = _clean_returns(returns)
    if len(arr) < 2:
        return None
    std_val = float(np.std(arr, ddof=1))
    if abs(std_val) <= 1e-15:
        return 0.0
    return float(std_val * math.sqrt(float(bars_per_year)))


def calculate_sharpe_ratio(
    returns: pd.Series | np.ndarray | list[float] | None,
    bars_per_year: int,
    risk_free_rate_annual: float = 0.0,
) -> float | None:
    """
    Tính hệ số Sharpe thường niên hóa:
        excess_return_t = return_t - rf_bar
        sharpe_annual = (mean(excess_return) / std(excess_return, ddof=1)) * sqrt(bars_per_year)
    Policy khi std == 0:
    - mean == 0 -> 0.0
    - mean > 0  -> +inf
    - mean < 0  -> -inf
    """
    rf_bar = annual_to_bar_rate(risk_free_rate_annual, bars_per_year)
    arr = _clean_returns(returns)
    if len(arr) < 2:
        return None
    excess = arr - rf_bar
    mean_ex = float(np.mean(excess))
    std_ex = float(np.std(excess, ddof=1))
    if std_ex <= 1e-12:
        if abs(mean_ex) <= 1e-12:
            return 0.0
        return float("inf") if mean_ex > 0.0 else float("-inf")
    return float((mean_ex / std_ex) * math.sqrt(float(bars_per_year)))


def calculate_downside_deviation(
    returns: pd.Series | np.ndarray | list[float] | None,
    bars_per_year: int = 1,
    risk_free_rate_annual: float = 0.0,
    *,
    annualize: bool = True,
) -> float | None:
    """
    Tính Downside Deviation theo chuẩn Lower Partial Moment bậc 2 (LPM2) trên toàn bộ N quan sát:
        downside_diff_t = min(0, return_t - rf_bar)
        downside_dev = sqrt(mean(downside_diff_t ** 2))
    """
    rf_bar = annual_to_bar_rate(risk_free_rate_annual, bars_per_year)
    arr = _clean_returns(returns)
    if len(arr) < 1:
        return None
    downside_diff = np.minimum(0.0, arr - rf_bar)
    dd_bar = float(math.sqrt(float(np.mean(np.square(downside_diff)))))
    if abs(dd_bar) <= 1e-15:
        return 0.0
    if annualize:
        return float(dd_bar * math.sqrt(float(bars_per_year)))
    return dd_bar


def calculate_sortino_ratio(
    returns: pd.Series | np.ndarray | list[float] | None,
    bars_per_year: int,
    risk_free_rate_annual: float = 0.0,
) -> float | None:
    """
    Tính hệ số Sortino thường niên hóa dựa trên LPM2:
        sortino_annual = (mean(return - rf_bar) / downside_deviation_bar) * sqrt(bars_per_year)
    """
    rf_bar = annual_to_bar_rate(risk_free_rate_annual, bars_per_year)
    arr = _clean_returns(returns)
    if len(arr) < 2:
        return None
    excess = arr - rf_bar
    mean_ex = float(np.mean(excess))
    downside_diff = np.minimum(0.0, excess)
    dd_bar = float(math.sqrt(float(np.mean(np.square(downside_diff)))))
    if dd_bar <= 1e-12:
        if abs(mean_ex) <= 1e-12:
            return 0.0
        return float("inf") if mean_ex > 0.0 else float("-inf")
    return float((mean_ex / dd_bar) * math.sqrt(float(bars_per_year)))


def calculate_calmar_ratio(
    cagr_pct: float | None,
    max_drawdown_pct: float | None,
) -> float | None:
    """
    Tính hệ số Calmar:
        calmar_ratio = cagr_pct / abs(max_drawdown_pct)
    Chỉ tính khi CAGR hợp lệ và Max Drawdown khác 0 (nếu Max Drawdown == 0 trả về None).
    """
    if cagr_pct is None or max_drawdown_pct is None:
        return None
    if not math.isfinite(float(cagr_pct)) or not math.isfinite(float(max_drawdown_pct)):
        return None
    dd_abs = abs(float(max_drawdown_pct))
    if dd_abs <= 1e-12:
        return None
    return float(float(cagr_pct) / dd_abs)


def calculate_return_distribution_metrics(
    returns: pd.Series | np.ndarray | list[float] | None,
) -> dict[str, float | None]:
    """
    Tính các chỉ số phân phối lợi suất:
    - return_skewness: Sample Skewness (Fisher-Pearson, yêu cầu N >= 3).
    - return_kurtosis / return_excess_kurtosis: Excess Kurtosis theo chuẩn Fisher (phân phối chuẩn = 0.0, yêu cầu N >= 4).
    - tail_ratio: abs(percentile(returns, 95)) / abs(percentile(returns, 5)) (yêu cầu N >= 5).
    """
    arr = _clean_returns(returns)
    n = len(arr)
    s = pd.Series(arr, dtype=float)

    skew_val: float | None = None
    kurt_val: float | None = None
    tail_ratio: float | None = None

    if n >= 3 and float(np.std(arr, ddof=1)) > 1e-12:
        raw_skew = s.skew()
        if pd_valid(raw_skew):
            skew_val = float(raw_skew)

    if n >= 4 and float(np.std(arr, ddof=1)) > 1e-12:
        raw_kurt = s.kurt()
        if pd_valid(raw_kurt):
            kurt_val = float(raw_kurt)

    if n >= 5:
        p95 = float(np.percentile(arr, 95))
        p5 = float(np.percentile(arr, 5))
        if abs(p5) > 1e-12:
            tail_ratio = float(abs(p95) / abs(p5))
        elif abs(p95) > 1e-12:
            tail_ratio = float("inf")

    return {
        "return_skewness": skew_val,
        "return_kurtosis": kurt_val,
        "return_excess_kurtosis": kurt_val,
        "tail_ratio": tail_ratio,
    }
