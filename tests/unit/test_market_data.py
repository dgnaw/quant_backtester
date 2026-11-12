import pandas as pd
import pytest

from quant_backtester.data.market_data import MarketData


def test_market_data_creation_and_metadata_properties():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00:00+00:00", "2026-01-01 00:15:00+00:00"]),
        "open": [2000.0, 2005.0],
        "high": [2010.0, 2015.0],
        "low": [1995.0, 2000.0],
        "close": [2005.0, 2010.0],
        "volume": [100.0, 120.0],
    })
    md = MarketData(
        data=df,
        symbol="XAUUSD",
        timeframe="M15",
        source="EXNESS_SQLITE",
        price_type="BID",
        timezone="UTC",
        volume_type="tick_volume",
    )

    assert md.symbol == "XAUUSD"
    assert md.timeframe == "M15"
    assert md.price_type == "BID"
    assert md.start_time == pd.Timestamp("2026-01-01 00:00:00+00:00")
    assert md.end_time == pd.Timestamp("2026-01-01 00:15:00+00:00")
    assert len(md) == 2


def test_market_data_rejects_invalid_metadata():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00:00+00:00"]),
        "open": [2000.0],
        "high": [2010.0],
        "low": [1995.0],
        "close": [2005.0],
        "volume": [100.0],
    })
    with pytest.raises(ValueError, match="price_type"):
        MarketData(data=df, symbol="XAUUSD", timeframe="M15", source="CSV", price_type="INVALID")

    with pytest.raises(ValueError, match="symbol"):
        MarketData(data=df, symbol="", timeframe="M15", source="CSV")

