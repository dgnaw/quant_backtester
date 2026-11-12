from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from .time_utils import parse_timeframe_delta


class GapType(Enum):
    MISSING_CANDLE = "MISSING_CANDLE"
    WEEKEND_GAP = "WEEKEND_GAP"
    HOLIDAY_GAP = "HOLIDAY_GAP"
    SESSION_GAP = "SESSION_GAP"
    UNKNOWN_GAP = "UNKNOWN_GAP"


@dataclass
class TimeGap:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    gap_type: GapType
    missing_count: int
    missing_timestamps: list[pd.Timestamp] = field(default_factory=list)


class TradingCalendar:
    """
    Xác định thời gian giao dịch hợp lệ của thị trường Forex / XAUUSD.
    Có thể cấu hình giờ đóng cửa cuối tuần, giờ nghỉ bảo trì hàng ngày (daily break) và ngày lễ.
    """
    def __init__(
        self,
        friday_close_hour: int = 22,
        sunday_open_hour: int = 24,
        daily_break_start_hour: int | None = None,
        daily_break_end_hour: int | None = None,
        holidays: set[str] | None = None,
    ):
        self.friday_close_hour = friday_close_hour
        self.sunday_open_hour = sunday_open_hour
        self.daily_break_start_hour = daily_break_start_hour
        self.daily_break_end_hour = daily_break_end_hour
        self.holidays = holidays if holidays is not None else set()

    def is_weekend(self, ts: pd.Timestamp) -> bool:
        wd = ts.weekday()  # Monday=0 ... Friday=4, Saturday=5, Sunday=6
        if wd == 5:
            return True
        if wd == 4 and ts.hour >= self.friday_close_hour:
            return True
        if wd == 6 and ts.hour < self.sunday_open_hour:
            return True
        return False

    def is_holiday(self, ts: pd.Timestamp) -> bool:
        date_str = ts.strftime("%Y-%m-%d")
        return date_str in self.holidays

    def is_daily_break(self, ts: pd.Timestamp) -> bool:
        if self.daily_break_start_hour is None or self.daily_break_end_hour is None:
            return False
        return self.daily_break_start_hour <= ts.hour < self.daily_break_end_hour


class MissingCandleDetector:
    """
    Phát hiện và phân loại các khoảng trống thời gian (Gaps) giữa các nến.
    Tuyệt đối KHÔNG tự động chèn nến giả vào dữ liệu gốc.
    """
    def __init__(self, calendar: TradingCalendar | None = None):
        self.calendar = calendar if calendar is not None else TradingCalendar()

    def detect_gaps(self, data: pd.DataFrame, timeframe: str = "M15") -> list[TimeGap]:
        if len(data) < 2:
            return []

        delta = parse_timeframe_delta(timeframe)
        times = pd.to_datetime(data["time"]).reset_index(drop=True)
        gaps: list[TimeGap] = []

        for i in range(1, len(times)):
            prev_t = times.iloc[i - 1]
            curr_t = times.iloc[i]
            diff = curr_t - prev_t

            if diff <= delta:
                continue

            # Tạo danh sách các mốc nến bị khuyết giữa prev_t và curr_t
            expected_slots: list[pd.Timestamp] = []
            cursor = prev_t + delta
            while cursor < curr_t:
                expected_slots.append(cursor)
                cursor += delta

            if not expected_slots:
                continue

            # Phân loại nguyên nhân của các nến khuyết
            if all(self.calendar.is_weekend(t) for t in expected_slots):
                gaps.append(
                    TimeGap(
                        start_time=prev_t,
                        end_time=curr_t,
                        gap_type=GapType.WEEKEND_GAP,
                        missing_count=0,
                        missing_timestamps=[],
                    )
                )
            elif all(self.calendar.is_holiday(t) or self.calendar.is_weekend(t) for t in expected_slots):
                gaps.append(
                    TimeGap(
                        start_time=prev_t,
                        end_time=curr_t,
                        gap_type=GapType.HOLIDAY_GAP,
                        missing_count=0,
                        missing_timestamps=[],
                    )
                )
            elif all(self.calendar.is_daily_break(t) for t in expected_slots):
                gaps.append(
                    TimeGap(
                        start_time=prev_t,
                        end_time=curr_t,
                        gap_type=GapType.SESSION_GAP,
                        missing_count=0,
                        missing_timestamps=[],
                    )
                )
            else:
                # Lọc ra những mốc thực sự nằm trong phiên giao dịch mà lại bị mất nến
                truly_missing = [
                    t
                    for t in expected_slots
                    if not (
                        self.calendar.is_weekend(t)
                        or self.calendar.is_holiday(t)
                        or self.calendar.is_daily_break(t)
                    )
                ]
                if truly_missing:
                    gaps.append(
                        TimeGap(
                            start_time=prev_t,
                            end_time=curr_t,
                            gap_type=GapType.MISSING_CANDLE,
                            missing_count=len(truly_missing),
                            missing_timestamps=truly_missing,
                        )
                    )
                else:
                    gaps.append(
                        TimeGap(
                            start_time=prev_t,
                            end_time=curr_t,
                            gap_type=GapType.WEEKEND_GAP,
                            missing_count=0,
                            missing_timestamps=[],
                        )
                    )

        return gaps
