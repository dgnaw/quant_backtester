import sqlite3
import pandas as pd
import pytest

from quant_backtester.data.loader import DataLoader
from quant_backtester.data.time_utils import TimeConfig


def test_data_loader_from_csv(tmp_path):
    csv_file = tmp_path / "xauusd_m15.csv"
    df_raw = pd.DataFrame({
        "time": ["2026-01-01 07:00:00", "2026-01-01 07:15:00", "2026-01-01 07:30:00"],
        "Open": [2000.0, 2005.0, 2010.0],
        "High": [2010.0, 2015.0, 2020.0],
        "Low": [1995.0, 2000.0, 2005.0],
        "Close": [2005.0, 2010.0, 2015.0],
        "tick_volume": [100, 110, 120],
    })
    df_raw.to_csv(csv_file, index=False)

    loader = DataLoader(time_config=TimeConfig(source_timezone="Asia/Ho_Chi_Minh", target_timezone="UTC"))
    md = loader.load_csv(csv_file, symbol="XAUUSD", timeframe="M15", price_type="BID")

    assert md.symbol == "XAUUSD"
    assert md.timeframe == "M15"
    assert md.volume_type == "tick_volume"
    assert md.timezone == "UTC"
    assert md.start_time == pd.Timestamp("2026-01-01 00:00:00+00:00")
    assert list(md.data.columns) == ["time", "open", "high", "low", "close", "volume"]


def test_data_loader_from_sqlite(tmp_path):
    db_file = tmp_path / "test_market.db"
    conn = sqlite3.connect(db_file)
    df_raw = pd.DataFrame({
        "time": ["2026-01-01 00:00:00+00:00", "2026-01-01 00:15:00+00:00"],
        "open": [1.1000, 1.1010],
        "high": [1.1020, 1.1030],
        "low": [1.0990, 1.1000],
        "close": [1.1010, 1.1020],
        "tick_volume": [500, 600],
        "spread": [10, 10],
    })
    df_raw.to_sql("EURUSD_M15", conn, index=False)
    conn.close()

    loader = DataLoader()
    md = loader.load_sqlite(db_file, table_name="EURUSD_M15", symbol="EURUSD", timeframe="M15")

    assert md.symbol == "EURUSD"
    assert len(md) == 2
    assert md.data["close"].iloc[-1] == pytest.approx(1.1020)


def test_data_loader_rejects_corrupted_data(tmp_path):
    csv_file = tmp_path / "bad.csv"
    df_bad = pd.DataFrame({
        "time": ["2026-01-01 00:00:00", "2026-01-01 00:15:00"],
        "open": [2000.0, 2005.0],
        "high": [1900.0, 2015.0],  # High < Open!
        "low": [1995.0, 2000.0],
        "close": [2005.0, 2010.0],
    })
    df_bad.to_csv(csv_file, index=False)

    loader = DataLoader()
    with pytest.raises(ValueError, match="Dữ liệu không hợp lệ"):
        loader.load_csv(csv_file, symbol="XAUUSD", timeframe="M15")

