from __future__ import annotations
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import pandas as pd


TIMEFRAME_DELTAS: dict[str, pd.Timedelta] = {
    "M1": pd.Timedelta(minutes=1),
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}


@dataclass
class TimeConfig:
    source_timezone: str | None = "UTC"
    target_timezone: str = "UTC"


def parse_timeframe_delta(timeframe: str) -> pd.Timedelta:
    tf = str(timeframe).strip().upper()
    if tf not in TIMEFRAME_DELTAS:
        raise ValueError(f"Timeframe '{timeframe}' không được hỗ trợ. Hỗ trợ: {list(TIMEFRAME_DELTAS.keys())}")
    return TIMEFRAME_DELTAS[tf]


def normalize_timestamps(series: pd.Series, config: TimeConfig | None = None) -> pd.Series:
    """
    Chuẩn hóa chuỗi thời gian về target_timezone (mặc định là UTC) có xử lý DST qua ZoneInfo.
    Nếu dữ liệu đầu vào là naive (chưa có timezone), bắt buộc phải chỉ định rõ source_timezone.
    """
    cfg = config if config is not None else TimeConfig()
    parsed = pd.to_datetime(series, errors="raise")

    target_tz = ZoneInfo(cfg.target_timezone)

    if parsed.dt.tz is None:
        if not cfg.source_timezone:
            raise ValueError("Timestamp đầu vào chưa có timezone; bắt buộc phải cấu hình rõ source_timezone.")
        source_tz = ZoneInfo(cfg.source_timezone)
        localized = parsed.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        if localized.isna().any():
            raise ValueError("Phát hiện timestamp không tồn tại hoặc nhập nhằng khi chuyển đổi DST.")
        return localized.dt.tz_convert(target_tz)

    return parsed.dt.tz_convert(target_tz)

