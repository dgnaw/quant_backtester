import pandas as pd
import pytest

from quant_backtester.data.market_data import MarketData
from quant_backtester.data.multi_asset_feed import MultiAssetDataFeed


def test_multi_asset_and_multi_timeframe_synchronization_no_lookahead():
    # M15 bars từ 09:00 đến 11:00
    m15_times = pd.date_range("2026-01-01 09:00:00+00:00", "2026-01-01 11:00:00+00:00", freq="15min")
    df_m15 = pd.DataFrame({
        "time": m15_times,
        "open": [2000.0 + i for i in range(len(m15_times))],
        "high": [2005.0 + i for i in range(len(m15_times))],
        "low": [1995.0 + i for i in range(len(m15_times))],
        "close": [2001.0 + i for i in range(len(m15_times))],
        "volume": [100.0] * len(m15_times),
    })
    md_xau_m15 = MarketData(data=df_m15, symbol="XAUUSD", timeframe="M15", source="TEST")

    # EURUSD M15 cùng khung giờ
    df_eur_m15 = pd.DataFrame({
        "time": m15_times,
        "open": [1.1000] * len(m15_times),
        "high": [1.1010] * len(m15_times),
        "low": [1.0990] * len(m15_times),
        "close": [1.1005] * len(m15_times),
        "volume": [200.0] * len(m15_times),
    })
    md_eur_m15 = MarketData(data=df_eur_m15, symbol="EURUSD", timeframe="M15", source="TEST")

    # H1 bars:
    # - Nến H1 09:00 (mở 09:00, đóng lúc 10:00) -> Close = 2050.0
    # - Nến H1 10:00 (mở 10:00, đóng lúc 11:00) -> Close = 2099.0
    h1_times = pd.to_datetime(["2026-01-01 09:00:00+00:00", "2026-01-01 10:00:00+00:00"])
    df_h1 = pd.DataFrame({
        "time": h1_times,
        "open": [2000.0, 2050.0],
        "high": [2060.0, 2100.0],
        "low": [1990.0, 2040.0],
        "close": [2050.0, 2099.0],
        "volume": [500.0, 600.0],
    })
    md_xau_h1 = MarketData(data=df_h1, symbol="XAUUSD", timeframe="H1", source="TEST")

    feed = MultiAssetDataFeed()
    feed.add_market_data(md_xau_m15)
    feed.add_market_data(md_eur_m15)
    feed.add_market_data(md_xau_h1)

    # Tiến tới thời điểm nến M15 mở lúc 10:15 (nến M15 10:15 kết thúc lúc 10:30)
    while feed.current_time < pd.Timestamp("2026-01-01 10:15:00+00:00"):
        feed.advance()

    assert feed.current_time == pd.Timestamp("2026-01-01 10:15:00+00:00")

    # Kiểm tra truy cập đồng thời nhiều symbol ở M15
    bar_xau = feed.get_current_bar("XAUUSD", timeframe="M15")
    bar_eur = feed.get_current_bar("EURUSD", timeframe="M15")
    assert bar_xau is not None and bar_eur is not None
    assert bar_xau["time"] == pd.Timestamp("2026-01-01 10:15:00+00:00")

    # QUAN TRỌNG: Tại thời điểm M15 = 10:15 (kết thúc 10:30):
    # - Nến H1 09:00 (đóng lúc 10:00) ĐÃ HOÀN THÀNH -> Được phép nhìn thấy (close = 2050.0)
    # - Nến H1 10:00 (đóng lúc 11:00) CHƯA HOÀN THÀNH -> TUYỆT ĐỐI KHÔNG ĐƯỢC NHÌN THẤY!
    h1_history = feed.get_history("XAUUSD", count=5, timeframe="H1")
    assert len(h1_history) == 1
    assert h1_history.iloc[-1]["time"] == pd.Timestamp("2026-01-01 09:00:00+00:00")
    assert h1_history.iloc[-1]["close"] == 2050.0

    # Tiến tới M15 = 10:45 (đóng lúc 11:00 -> lúc này nến H1 10:00–11:00 vừa đóng cửa xong!)
    while feed.current_time < pd.Timestamp("2026-01-01 10:45:00+00:00"):
        feed.advance()

    h1_history_at_11 = feed.get_history("XAUUSD", count=5, timeframe="H1")
    assert len(h1_history_at_11) == 2
    assert h1_history_at_11.iloc[-1]["time"] == pd.Timestamp("2026-01-01 10:00:00+00:00")
    assert h1_history_at_11.iloc[-1]["close"] == 2099.0

