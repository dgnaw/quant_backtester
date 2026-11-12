from pathlib import Path
import pandas as pd

from .schema import STANDARD_COLUMNS, REQUIRED_OHLC_COLUMNS, REQUIRED_COLUMNS, normalize_schema
from .validation_report import ValidationReport, ValidationIssue, IssueSeverity
from .validator import DataValidator, DataQualityPolicy, validate_ohlc
from .time_utils import TimeConfig, normalize_timestamps, parse_timeframe_delta, TIMEFRAME_DELTAS
from .trading_calendar import TradingCalendar, MissingCandleDetector, TimeGap, GapType
from .market_data import MarketData, VALID_PRICE_TYPES
from .loader import DataLoader
from .data_feed import DataFeed, SingleAssetDataFeed
from .multi_asset_feed import MultiAssetDataFeed


def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    return validate_ohlc(pd.read_csv(path))


__all__ = [
    "STANDARD_COLUMNS",
    "REQUIRED_OHLC_COLUMNS",
    "REQUIRED_COLUMNS",
    "normalize_schema",
    "ValidationReport",
    "ValidationIssue",
    "IssueSeverity",
    "DataValidator",
    "DataQualityPolicy",
    "validate_ohlc",
    "load_ohlc_csv",
    "TimeConfig",
    "normalize_timestamps",
    "parse_timeframe_delta",
    "TIMEFRAME_DELTAS",
    "TradingCalendar",
    "MissingCandleDetector",
    "TimeGap",
    "GapType",
    "MarketData",
    "VALID_PRICE_TYPES",
    "DataLoader",
    "DataFeed",
    "SingleAssetDataFeed",
    "MultiAssetDataFeed",
]
