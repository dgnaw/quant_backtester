from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class QuoteBar:
    """
    Chuẩn hóa thông tin nến Bid/Ask OHLC tại một thời điểm.
    Đảm bảo giá dương, hữu hạn, tính nhất quán của OHLC cho cả Bid và Ask, và Ask >= Bid tại từng trường.
    """
    symbol: str
    time: pd.Timestamp
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float

    def __post_init__(self):
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("QuoteBar.symbol phải là chuỗi không rỗng.")
        if self.time is None or pd.isna(self.time):
            raise ValueError("QuoteBar.time không hợp lệ.")
        object.__setattr__(self, "time", pd.Timestamp(self.time))

        fields = (
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
        )
        for name in fields:
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"QuoteBar.{name} phải là số thực.")
            f_val = float(val)
            if not math.isfinite(f_val) or f_val <= 0.0:
                raise ValueError(f"QuoteBar.{name} phải hữu hạn và > 0 (nhận {val}).")
            object.__setattr__(self, name, f_val)

        # Kiểm tra tính nhất quán OHLC của Bid
        if self.bid_high < max(self.bid_open, self.bid_close, self.bid_low) - 1e-12:
            raise ValueError("QuoteBar: bid_high phải >= bid_open, bid_close, bid_low (High không hợp lệ).")
        if self.bid_low > min(self.bid_open, self.bid_close, self.bid_high) + 1e-12:
            raise ValueError("QuoteBar: bid_low phải <= bid_open, bid_close, bid_high (Low không hợp lệ).")

        # Kiểm tra tính nhất quán OHLC của Ask
        if self.ask_high < max(self.ask_open, self.ask_close, self.ask_low) - 1e-12:
            raise ValueError("QuoteBar: ask_high phải >= ask_open, ask_close, ask_low (High không hợp lệ).")
        if self.ask_low > min(self.ask_open, self.ask_close, self.ask_high) + 1e-12:
            raise ValueError("QuoteBar: ask_low phải <= ask_open, ask_close, ask_high (Low không hợp lệ).")

        # Kiểm tra Ask >= Bid tại từng trường tương ứng
        if (
            self.ask_open < self.bid_open - 1e-12
            or self.ask_high < self.bid_high - 1e-12
            or self.ask_low < self.bid_low - 1e-12
            or self.ask_close < self.bid_close - 1e-12
        ):
            raise ValueError("QuoteBar: Ask phải >= Bid tại mọi trường Open, High, Low, Close.")

    @classmethod
    def from_series(cls, row: pd.Series, symbol: str = "XAUUSD") -> QuoteBar:
        return cls(
            symbol=symbol,
            time=pd.Timestamp(row["time"]),
            bid_open=float(row["bid_open"]),
            bid_high=float(row["bid_high"]),
            bid_low=float(row["bid_low"]),
            bid_close=float(row["bid_close"]),
            ask_open=float(row["ask_open"]),
            ask_high=float(row["ask_high"]),
            ask_low=float(row["ask_low"]),
            ask_close=float(row["ask_close"]),
        )

    @property
    def mid_open(self) -> float:
        return (self.bid_open + self.ask_open) / 2.0

    @property
    def mid_high(self) -> float:
        return (self.bid_high + self.ask_high) / 2.0

    @property
    def mid_low(self) -> float:
        return (self.bid_low + self.ask_low) / 2.0

    @property
    def mid_close(self) -> float:
        return (self.bid_close + self.ask_close) / 2.0

    @property
    def spread_open(self) -> float:
        return self.ask_open - self.bid_open

    @property
    def spread_high(self) -> float:
        return self.ask_high - self.bid_high

    @property
    def spread_low(self) -> float:
        return self.ask_low - self.bid_low

    @property
    def spread_close(self) -> float:
        return self.ask_close - self.bid_close
