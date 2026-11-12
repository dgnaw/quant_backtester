import pandas as pd
import pytest

from quant_backtester.data.time_utils import TimeConfig, normalize_timestamps
from quant_backtester.data.trading_calendar import (
    TradingCalendar,
    GapType,
    TimeGap,
    MissingCandleDetector,
)


def test_timezone_normalization_from_ho_chi_minh_to_utc():
    s = pd.Series(["2026-11-10 09:00:00", "2026-11-10 09:15:00"])
    cfg = TimeConfig(source_timezone="Asia/Ho_Chi_Minh", target_timezone="UTC")
    utc_s = normalize_timestamps(s, cfg)

    assert str(utc_s.dt.tz) == "UTC"
    assert utc_s.iloc[0] == pd.Timestamp("2026-11-10 02:00:00", tz="UTC")
    assert utc_s.iloc[1] == pd.Timestamp("2026-11-10 02:15:00", tz="UTC")


def test_dst_transition_new_york_to_utc():
    # US Summer EDT (UTC-4) vs Winter EST (UTC-5)
    s = pd.Series(["2026-07-15 10:00:00", "2026-12-15 10:00:00"])
    cfg = TimeConfig(source_timezone="America/New_York", target_timezone="UTC")
    utc_s = normalize_timestamps(s, cfg)

    # July is EDT (UTC-4) -> 14:00 UTC; December is EST (UTC-5) -> 15:00 UTC
    assert utc_s.iloc[0] == pd.Timestamp("2026-07-15 14:00:00", tz="UTC")
    assert utc_s.iloc[1] == pd.Timestamp("2026-12-15 15:00:00", tz="UTC")


def test_naive_timestamp_requires_explicit_source_timezone():
    s = pd.Series(["2026-11-10 09:00:00"])
    cfg = TimeConfig(source_timezone=None, target_timezone="UTC")
    with pytest.raises(ValueError, match="source_timezone"):
        normalize_timestamps(s, cfg)


def test_missing_candle_detection_within_session():
    # Wednesday 2026-01-07: 10:00, 10:15, 10:30, 10:45, [11:00 missing], 11:15, 11:30
    times = pd.to_datetime([
        "2026-01-07 10:00:00+00:00",
        "2026-01-07 10:15:00+00:00",
        "2026-01-07 10:30:00+00:00",
        "2026-01-07 10:45:00+00:00",
        "2026-01-07 11:15:00+00:00",
        "2026-01-07 11:30:00+00:00",
    ])
    df = pd.DataFrame({
        "time": times,
        "open": [2000.0] * 6,
        "high": [2005.0] * 6,
        "low": [1995.0] * 6,
        "close": [2001.0] * 6,
        "volume": [100.0] * 6,
    })

    calendar = TradingCalendar()
    detector = MissingCandleDetector(calendar=calendar)
    gaps = detector.detect_gaps(df, timeframe="M15")

    assert len(gaps) == 1
    assert gaps[0].gap_type == GapType.MISSING_CANDLE
    assert gaps[0].missing_count == 1
    assert pd.Timestamp("2026-01-07 11:00:00+00:00") in gaps[0].missing_timestamps
    # Ensure input df is not mutated with fake candles
    assert len(df) == 6


def test_weekend_and_holiday_and_session_gaps_are_classified_valid():
    # Friday 2026-01-09 21:45 UTC -> Monday 2026-01-12 00:00 UTC (Weekend Gap)
    # Plus a daily maintenance break 22:00 -> 23:00 UTC on Monday (Session Gap)
    # Plus a holiday on 2026-01-13 (Holiday Gap)
    times = pd.to_datetime([
        "2026-01-09 21:45:00+00:00",  # Friday last bar before weekend (22:00)
        "2026-01-12 00:00:00+00:00",  # Monday open (Weekend Gap)
        "2026-01-12 21:45:00+00:00",  # Before daily break (Missing candles between 00:00 and 21:45)
        "2026-01-12 23:00:00+00:00",  # After daily break 22:00-23:00 (Session Gap)
        "2026-01-12 23:15:00+00:00",
        "2026-01-12 23:30:00+00:00",
        "2026-01-12 23:45:00+00:00",  # Before holiday 2026-01-13
        "2026-01-14 00:00:00+00:00",  # After holiday (Holiday Gap)
    ])
    df = pd.DataFrame({
        "time": times,
        "open": [2000.0] * len(times),
        "high": [2005.0] * len(times),
        "low": [1995.0] * len(times),
        "close": [2001.0] * len(times),
        "volume": [100.0] * len(times),
    })

    calendar = TradingCalendar(
        daily_break_start_hour=22,
        daily_break_end_hour=23,
        holidays={"2026-01-13"},
    )
    detector = MissingCandleDetector(calendar=calendar)
    gaps = detector.detect_gaps(df, timeframe="M15")

    gap_types = [g.gap_type for g in gaps]
    assert GapType.WEEKEND_GAP in gap_types
    assert GapType.SESSION_GAP in gap_types
    assert GapType.HOLIDAY_GAP in gap_types
    # Only the gap from 00:00 to 21:45 on Monday is a missing candle gap
    missing_gaps = [g for g in gaps if g.gap_type == GapType.MISSING_CANDLE]
    assert len(missing_gaps) == 1
