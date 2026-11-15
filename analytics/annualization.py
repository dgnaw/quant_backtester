from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

from quant_backtester.data.trading_calendar import TradingCalendar


TIMEFRAME_MINUTES_MAP: dict[str, int] = {
    "M1": 1,
    "1M": 1,
    "1MIN": 1,
    "M5": 5,
    "5M": 5,
    "5MIN": 5,
    "M15": 15,
    "15M": 15,
    "15MIN": 15,
    "M30": 30,
    "30M": 30,
    "30MIN": 30,
    "H1": 60,
    "1H": 60,
    "60M": 60,
    "H4": 240,
    "4H": 240,
    "240M": 240,
    "D1": 1440,
    "1D": 1440,
    "D": 1440,
    "W1": 10080,
    "1W": 10080,
    "W": 10080,
}

DEFAULT_FOREX_BARS_PER_YEAR = 96 * 252  # 24,192 (M15, 24h x 252 ngày)


def bars_per_year_from_timeframe(
    timeframe: str,
    trading_days_per_year: int = 252,
    trading_hours_per_day: float = 24.0,
    trading_calendar: TradingCalendar | None = None,
) -> int:
    """
    Tính số Bar mỗi năm từ mã Timeframe (M1, M5, M15, M30, H1, H4, D1, W1)
    kết hợp với số ngày giao dịch trong năm và giờ giao dịch mỗi ngày (hoặc TradingCalendar).
    """
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe không được rỗng.")
    tf_key = timeframe.strip().upper()
    if tf_key not in TIMEFRAME_MINUTES_MAP:
        raise ValueError(
            f"Timeframe '{timeframe}' không được hỗ trợ. Hỗ trợ: {sorted(set(TIMEFRAME_MINUTES_MAP.keys()))}"
        )
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year phải > 0.")

    eff_hours = float(trading_hours_per_day)
    if trading_calendar is not None:
        if (
            trading_calendar.daily_break_start_hour is not None
            and trading_calendar.daily_break_end_hour is not None
            and trading_calendar.daily_break_end_hour > trading_calendar.daily_break_start_hour
        ):
            break_hours = float(
                trading_calendar.daily_break_end_hour - trading_calendar.daily_break_start_hour
            )
            eff_hours = max(1.0, 24.0 - break_hours)

    if eff_hours <= 0.0 or eff_hours > 24.0:
        raise ValueError("trading_hours_per_day phải nằm trong (0.0, 24.0].")

    if tf_key in {"W1", "1W", "W"}:
        return 52
    if tf_key in {"D1", "1D", "D"}:
        return int(trading_days_per_year)

    minutes = TIMEFRAME_MINUTES_MAP[tf_key]
    bars_per_day = (eff_hours * 60.0) / float(minutes)
    result = int(round(bars_per_day * float(trading_days_per_year)))
    if result <= 0:
        raise ValueError("bars_per_year tính được phải > 0.")
    return result


def infer_bars_per_year_from_timestamps(
    time_series: pd.Series | pd.Index | list[Any],
    trading_days_per_year: int = 252,
    trading_hours_per_day: float = 24.0,
    trading_calendar: TradingCalendar | None = None,
) -> int:
    """
    Suy luận số Bar mỗi năm từ khoảng cách thời gian trung vị (median delta) của chuỗi Timestamp.
    Loại trừ ảnh hưởng của các khoảng nghỉ cuối tuần/ngày lễ nhờ dùng median thay vì mean.
    """
    times = pd.to_datetime(pd.Series(list(time_series)))
    if len(times) < 2:
        raise ValueError(
            "Cần ít nhất 2 mốc thời gian để suy luận bars_per_year từ dữ liệu."
        )
    if not times.is_monotonic_increasing:
        raise ValueError("Chuỗi thời gian phải tăng dần để suy luận bars_per_year.")

    diffs = times.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        raise ValueError("Không tìm thấy khoảng cách thời gian dương giữa các quan sát.")

    median_delta = positive_diffs.median()
    delta_minutes = median_delta.total_seconds() / 60.0
    if delta_minutes <= 0.0:
        raise ValueError("Khoảng cách thời gian trung vị không hợp lệ.")

    if delta_minutes >= 5.0 * 1440.0:
        return 52
    if delta_minutes >= 1440.0 - 1e-6:
        return int(trading_days_per_year)

    eff_hours = float(trading_hours_per_day)
    if trading_calendar is not None:
        if (
            trading_calendar.daily_break_start_hour is not None
            and trading_calendar.daily_break_end_hour is not None
            and trading_calendar.daily_break_end_hour > trading_calendar.daily_break_start_hour
        ):
            break_hours = float(
                trading_calendar.daily_break_end_hour - trading_calendar.daily_break_start_hour
            )
            eff_hours = max(1.0, 24.0 - break_hours)

    bars_per_day = (eff_hours * 60.0) / delta_minutes
    inferred = int(round(bars_per_day * float(trading_days_per_year)))
    if inferred <= 0:
        raise ValueError("bars_per_year suy luận được không hợp lệ (<= 0).")
    return inferred


def resolve_bars_per_year(
    config: Any = None,
    timeframe: str | None = None,
    time_series: pd.Series | pd.Index | list[Any] | None = None,
    trading_calendar: TradingCalendar | None = None,
    *,
    auto_annualize_from_data: bool | None = None,
    trading_days_per_year: int | None = None,
    trading_hours_per_day: float | None = None,
) -> int:
    """
    Giải quyết hệ số thường niên hóa (bars_per_year) theo thứ tự ưu tiên:
    1. Nếu truyền trực tiếp `timeframe` (hoặc `config.timeframe` được đặt tường minh) -> tính theo Timeframe & TradingCalendar.
    2. Nếu bật `auto_annualize_from_data=True` và có `time_series` -> suy luận từ Timestamp.
    3. Nếu `config.bars_per_year` hợp lệ (> 0) -> dùng `config.bars_per_year`.
    4. Nếu không có config nhưng có `time_series` (len >= 2) -> suy luận từ Timestamp.
    5. Nếu không đủ dữ liệu -> ném ValueError tường minh.
    """
    analytics_cfg = getattr(config, "analytics", None) if config is not None else None
    eff_days = (
        int(trading_days_per_year)
        if trading_days_per_year is not None
        else int(getattr(analytics_cfg, "trading_days_per_year", 252))
    )
    eff_hours = (
        float(trading_hours_per_day)
        if trading_hours_per_day is not None
        else float(getattr(analytics_cfg, "trading_hours_per_day", 24.0))
    )
    eff_tf = timeframe
    if eff_tf is None and config is not None:
        eff_tf = getattr(config, "timeframe", None) or getattr(analytics_cfg, "timeframe", None)

    eff_auto = auto_annualize_from_data
    if eff_auto is None and config is not None:
        eff_auto = getattr(config, "auto_annualize_from_data", None)
        if eff_auto is None and analytics_cfg is not None:
            eff_auto = getattr(analytics_cfg, "auto_annualize_from_data", False)

    # Nếu config.bars_per_year được người dùng truyền khác mặc định và không truyền timeframe trực tiếp
    cfg_bars = getattr(config, "bars_per_year", None) if config is not None else None
    if cfg_bars is not None:
        if isinstance(cfg_bars, bool) or not isinstance(cfg_bars, (int, float)) or cfg_bars <= 0:
            raise ValueError("config.bars_per_year phải lớn hơn 0.")

    if timeframe is not None:
        return bars_per_year_from_timeframe(
            timeframe,
            trading_days_per_year=eff_days,
            trading_hours_per_day=eff_hours,
            trading_calendar=trading_calendar,
        )

    if eff_tf is not None:
        return bars_per_year_from_timeframe(
            str(eff_tf),
            trading_days_per_year=eff_days,
            trading_hours_per_day=eff_hours,
            trading_calendar=trading_calendar,
        )

    if bool(eff_auto) and time_series is not None:
        return infer_bars_per_year_from_timestamps(
            time_series,
            trading_days_per_year=eff_days,
            trading_hours_per_day=eff_hours,
            trading_calendar=trading_calendar,
        )

    if cfg_bars is not None:
        return int(cfg_bars)

    if time_series is not None and len(time_series) >= 2:
        return infer_bars_per_year_from_timestamps(
            time_series,
            trading_days_per_year=eff_days,
            trading_hours_per_day=eff_hours,
            trading_calendar=trading_calendar,
        )

    raise ValueError(
        "Không đủ thông tin để xác định bars_per_year (cần cung cấp config.bars_per_year, timeframe, hoặc time_series)."
    )
