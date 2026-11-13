import pandas as pd
import pytest

from quant_backtester.config import InstrumentConfig
from quant_backtester.execution.spread import FixedSpreadModel, ZeroSpreadModel


def _sample_ohlc() -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=2, freq="15min"),
        "open": [2000.0, 2010.0],
        "high": [2005.0, 2015.0],
        "low": [1995.0, 2005.0],
        "close": [2002.0, 2012.0],
    })


def test_fixed_spread_with_mid_bid_ask():
    df = _sample_ohlc()
    model = FixedSpreadModel(spread_price=0.40)

    # 1. MID: Bid = Mid - 0.20, Ask = Mid + 0.20
    q_mid = model.to_quote_frame(df, price_type="MID", symbol="XAUUSD")
    qb0 = model.get_quote_bar(q_mid, 0, symbol="XAUUSD")
    assert qb0.bid_open == pytest.approx(1999.80)
    assert qb0.ask_open == pytest.approx(2000.20)
    assert qb0.bid_high == pytest.approx(2004.80)
    assert qb0.ask_high == pytest.approx(2005.20)

    # 2. BID: Bid = Bid, Ask = Bid + 0.40
    q_bid = model.to_quote_frame(df, price_type="BID", symbol="XAUUSD")
    qb_bid = model.get_quote_bar(q_bid, 0, symbol="XAUUSD")
    assert qb_bid.bid_open == pytest.approx(2000.00)
    assert qb_bid.ask_open == pytest.approx(2000.40)

    # 3. ASK: Ask = Ask, Bid = Ask - 0.40
    q_ask = model.to_quote_frame(df, price_type="ASK", symbol="XAUUSD")
    qb_ask = model.get_quote_bar(q_ask, 0, symbol="XAUUSD")
    assert qb_ask.ask_open == pytest.approx(2000.00)
    assert qb_ask.bid_open == pytest.approx(1999.60)


def test_spread_from_points_and_zero_spread():
    df = _sample_ohlc()
    inst = InstrumentConfig(symbol="XAUUSD", point_size=0.01)
    # 25 points * 0.01 = 0.25 price spread
    model = FixedSpreadModel.from_points(spread_points=25, point_size=inst.point_size)
    assert model.spread_price == pytest.approx(0.25)

    zero = ZeroSpreadModel()
    q_zero = zero.to_quote_frame(df, price_type="MID", symbol="XAUUSD")
    qb = zero.get_quote_bar(q_zero, 0, symbol="XAUUSD")
    assert qb.bid_open == pytest.approx(qb.ask_open)


def test_existing_bid_ask_columns_and_double_spread_detection():
    df_ba = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=1, freq="15min"),
        "open": [2000.0],
        "high": [2005.0],
        "low": [1995.0],
        "close": [2002.0],
        "bid_open": [1999.9],
        "bid_high": [2004.9],
        "bid_low": [1994.9],
        "bid_close": [2001.9],
        "ask_open": [2000.1],
        "ask_high": [2005.1],
        "ask_low": [1995.1],
        "ask_close": [2002.1],
    })

    # ZeroSpreadModel uses pre-existing Bid/Ask directly
    zero = ZeroSpreadModel()
    q_df = zero.to_quote_frame(df_ba, price_type="MID", symbol="XAUUSD")
    qb = zero.get_quote_bar(q_df, 0, symbol="XAUUSD")
    assert qb.bid_open == pytest.approx(1999.9)
    assert qb.ask_open == pytest.approx(2000.1)

    # FixedSpreadModel with spread > 0 on pre-existing Bid/Ask must raise DOUBLE_SPREAD
    nonzero = FixedSpreadModel(spread_price=0.20)
    with pytest.raises(ValueError, match="DOUBLE_SPREAD"):
        nonzero.to_quote_frame(df_ba, price_type="MID", symbol="XAUUSD")


def test_invalid_spread_rejected():
    with pytest.raises(ValueError):
        FixedSpreadModel(spread_price=-0.1)
    with pytest.raises(ValueError):
        FixedSpreadModel(spread_price=float("nan"))
    with pytest.raises(ValueError):
        FixedSpreadModel(spread_price=float("inf"))

