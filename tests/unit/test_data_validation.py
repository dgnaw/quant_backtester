import numpy as np
import pandas as pd
import pytest

from quant_backtester.data.schema import (
    STANDARD_COLUMNS,
    REQUIRED_OHLC_COLUMNS,
    normalize_schema,
)
from quant_backtester.data.validation_report import (
    ValidationReport,
    ValidationIssue,
    IssueSeverity,
)
from quant_backtester.data.validator import DataValidator, DataQualityPolicy


def make_valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:15:00", "2026-01-01 00:30:00"]),
        "open": [2000.0, 2005.0, 2010.0],
        "high": [2010.0, 2015.0, 2020.0],
        "low": [1995.0, 2000.0, 2005.0],
        "close": [2005.0, 2010.0, 2015.0],
        "volume": [100.0, 150.0, 120.0],
    })


def test_schema_normalization():
    raw = pd.DataFrame({
        "TIME": ["2026-01-01 00:00:00", "2026-01-01 00:15:00"],
        "Open": [100.0, 101.0],
        "High": [105.0, 106.0],
        "Low": [99.0, 100.0],
        "Close": [102.0, 104.0],
        "tick_volume": [50, 60],
    })
    norm_df, vol_type = normalize_schema(raw)
    assert list(norm_df.columns) == list(STANDARD_COLUMNS)
    assert vol_type == "tick_volume"
    assert norm_df["volume"].tolist() == [50.0, 60.0]


def test_validator_accepts_valid_data():
    df = make_valid_df()
    validator = DataValidator()
    report = validator.validate(df)

    assert isinstance(report, ValidationReport)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_rows == 3
    assert report.invalid_rows == 0


def test_validator_detects_missing_required_columns():
    df = make_valid_df().drop(columns=["close"])
    report = DataValidator().validate(df)
    assert report.is_valid is False
    assert any("close" in err.message.lower() for err in report.errors)


def test_validator_detects_nan_and_inf_and_non_positive():
    df = make_valid_df()
    df.loc[0, "open"] = np.nan
    df.loc[1, "high"] = np.inf
    df.loc[2, "low"] = -5.0

    report = DataValidator().validate(df)
    assert report.is_valid is False
    assert len(report.errors) >= 3
    assert report.invalid_rows == 3


def test_validator_detects_ohlc_inconsistency():
    df = make_valid_df()
    # High < Open (2000) and Low > Close
    df.loc[1, "high"] = 1990.0
    df.loc[2, "low"] = 2050.0

    report = DataValidator().validate(df)
    assert report.is_valid is False
    assert report.invalid_rows == 2
    assert any("ohlc" in err.code.lower() or "high" in err.message.lower() for err in report.errors)


def test_validator_detects_duplicate_and_unsorted_timestamps_without_mutating():
    df = make_valid_df()
    # Unsorted & duplicate
    df["time"] = pd.to_datetime(["2026-01-01 00:30:00", "2026-01-01 00:15:00", "2026-01-01 00:15:00"])
    original_times = df["time"].copy()

    report = DataValidator().validate(df)
    assert report.is_valid is False
    assert any("duplicate" in err.code.lower() or "trùng" in err.message.lower() for err in report.errors)
    assert any("unsorted" in err.code.lower() or "sắp xếp" in err.message.lower() for err in report.errors)
    # Validator must NOT mutate input DataFrame
    pd.testing.assert_series_equal(df["time"], original_times)


def test_data_quality_policy_volume_handling():
    df = make_valid_df()
    df.loc[0, "volume"] = 0.0

    # Default policy: zero volume is a warning, not an error
    report_default = DataValidator(DataQualityPolicy(allow_zero_volume=True)).validate(df)
    assert report_default.is_valid is True
    assert len(report_default.warnings) >= 1

    # Strict policy: zero volume is rejected as an error
    report_strict = DataValidator(DataQualityPolicy(allow_zero_volume=False)).validate(df)
    assert report_strict.is_valid is False
    assert len(report_strict.errors) >= 1

