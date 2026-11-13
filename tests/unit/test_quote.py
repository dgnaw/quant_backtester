import pandas as pd
import pytest

from quant_backtester.models.quote import QuoteBar


def test_quote_bar_valid_and_derived_properties():
    qb = QuoteBar(
        symbol="XAUUSD",
        time=pd.Timestamp("2026-01-01 00:00"),
        bid_open=1999.9,
        bid_high=2004.9,
        bid_low=1994.9,
        bid_close=2001.9,
        ask_open=2000.1,
        ask_high=2005.1,
        ask_low=1995.1,
        ask_close=2002.1,
    )
    assert qb.mid_open == pytest.approx(2000.0)
    assert qb.mid_high == pytest.approx(2005.0)
    assert qb.mid_low == pytest.approx(1995.0)
    assert qb.mid_close == pytest.approx(2002.0)

    assert qb.spread_open == pytest.approx(0.2)
    assert qb.spread_high == pytest.approx(0.2)
    assert qb.spread_low == pytest.approx(0.2)
    assert qb.spread_close == pytest.approx(0.2)

    # Frozen immutability
    with pytest.raises(Exception):
        qb.bid_open = 1000.0  # type: ignore


def test_quote_bar_rejects_invalid_ohlc_or_crossed_bid_ask():
    ts = pd.Timestamp("2026-01-01 00:00")

    # Ask < Bid
    with pytest.raises(ValueError, match="Ask"):
        QuoteBar(
            symbol="XAUUSD",
            time=ts,
            bid_open=2000.5,
            bid_high=2005.0,
            bid_low=1995.0,
            bid_close=2001.0,
            ask_open=2000.0,
            ask_high=2005.5,
            ask_low=1995.5,
            ask_close=2001.5,
        )

    # Bid High < Bid Open
    with pytest.raises(ValueError, match="High"):
        QuoteBar(
            symbol="XAUUSD",
            time=ts,
            bid_open=2000.0,
            bid_high=1998.0,
            bid_low=1995.0,
            bid_close=1997.0,
            ask_open=2000.2,
            ask_high=2005.2,
            ask_low=1995.2,
            ask_close=1997.2,
        )

    # Non-positive or NaN price
    with pytest.raises(ValueError):
        QuoteBar(
            symbol="XAUUSD",
            time=ts,
            bid_open=-1.0,
            bid_high=2005.0,
            bid_low=-5.0,
            bid_close=2000.0,
            ask_open=2000.0,
            ask_high=2005.2,
            ask_low=1995.2,
            ask_close=2000.2,
        )

