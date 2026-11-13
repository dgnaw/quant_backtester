from __future__ import annotations
from abc import ABC, abstractmethod
import math
import pandas as pd

from quant_backtester.models.quote import QuoteBar

BID_ASK_COLUMNS = (
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)


class SpreadModel(ABC):
    @property
    @abstractmethod
    def spread_price(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def to_quote_frame(self, df: pd.DataFrame, price_type: str = "MID", symbol: str = "XAUUSD") -> pd.DataFrame:
        raise NotImplementedError

    def get_quote_bar(self, quote_df: pd.DataFrame, bar_index: int, symbol: str = "XAUUSD") -> QuoteBar:
        row = quote_df.iloc[bar_index]
        return QuoteBar(
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


BaseSpreadModel = SpreadModel


class FixedSpreadModel(SpreadModel):
    def __init__(self, spread_price: float = 0.0):
        if isinstance(spread_price, bool) or not isinstance(spread_price, (int, float)):
            raise ValueError("spread_price phải là số thực.")
        s = float(spread_price)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError("spread_price phải hữu hạn và >= 0.")
        self._spread_price = s

    @classmethod
    def from_points(cls, spread_points: float, point_size: float) -> FixedSpreadModel:
        if point_size <= 0 or not math.isfinite(point_size):
            raise ValueError("point_size phải > 0.")
        if spread_points < 0 or not math.isfinite(spread_points):
            raise ValueError("spread_points phải >= 0.")
        return cls(spread_price=float(spread_points) * float(point_size))

    @property
    def spread_price(self) -> float:
        return self._spread_price

    def to_quote_frame(self, df: pd.DataFrame, price_type: str = "MID", symbol: str = "XAUUSD") -> pd.DataFrame:
        out = df.copy()
        has_bid_ask = all(c in out.columns for c in BID_ASK_COLUMNS)

        if has_bid_ask:
            if self._spread_price > 0.0:
                raise ValueError(
                    "DOUBLE_SPREAD: Dữ liệu đầu vào đã có sẵn các cột Bid/Ask OHLC, không thể áp dụng thêm Spread > 0."
                )
            return out

        pt = str(price_type).strip().upper()
        if pt not in ("MID", "BID", "ASK"):
            raise ValueError(f"price_type không hợp lệ: {price_type!r}. Chỉ hỗ trợ MID, BID, ASK.")

        s = self._spread_price
        half_s = s / 2.0

        for field in ("open", "high", "low", "close"):
            base_col = out[field].astype(float)
            if pt == "MID":
                out[f"bid_{field}"] = base_col - half_s
                out[f"ask_{field}"] = base_col + half_s
            elif pt == "BID":
                out[f"bid_{field}"] = base_col
                out[f"ask_{field}"] = base_col + s
            elif pt == "ASK":
                out[f"bid_{field}"] = base_col - s
                out[f"ask_{field}"] = base_col

        return out


class ZeroSpreadModel(FixedSpreadModel):
    def __init__(self):
        super().__init__(spread_price=0.0)

