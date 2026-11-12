import pandas as pd
import pytest

from quant_backtester.data.market_data import MarketData
from quant_backtester.data.data_feed import SingleAssetDataFeed


def make_market_data() -> MarketData:
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2026-01-01 00:00:00+00:00",
            "2026-01-01 00:15:00+00:00",
            "2026-01-01 00:30:00+00:00",
            "2026-01-01 00:45:00+00:00",
        ]),
        "open": [2000.0, 2005.0, 2010.0, 2015.0],
        "high": [2006.0, 2011.0, 2016.0, 2021.0],
        "low": [1999.0, 2004.0, 2009.0, 2014.0],
        "close": [2005.0, 2010.0, 2015.0, 2020.0],
        "volume": [100.0, 110.0, 120.0, 130.0],
    })
    return MarketData(data=df, symbol="XAUUSD", timeframe="M15", source="TEST")


def test_single_asset_data_feed_sequential_and_no_lookahead():
    md = make_market_data()
    feed = SingleAssetDataFeed(md)

    # Bar 0 (00:00)
    bar0 = feed.get_current_bar("XAUUSD")
    assert bar0["close"] == 2005.0
    hist0 = feed.get_history("XAUUSD", count=10)
    # Tại bar 0, history tối đa chỉ có 1 dòng (không bao giờ được lấy sang bar 1, 2, 3)
    assert len(hist0) == 1
    assert hist0.iloc[-1]["time"] == pd.Timestamp("2026-01-01 00:00:00+00:00")

    # Advance sang Bar 1 (00:15)
    assert feed.advance() is True
    bar1 = feed.get_current_bar("XAUUSD")
    assert bar1["close"] == 2010.0
    hist1 = feed.get_history("XAUUSD", count=2)
    assert len(hist1) == 2
    assert hist1["close"].tolist() == [2005.0, 2010.0]

    # Advance đến hết
    assert feed.advance() is True  # Bar 2
    assert feed.advance() is True  # Bar 3
    assert feed.advance() is False  # Hết dữ liệu

