from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

from .market_data import MarketData


class DataFeed(ABC):
    """
    Interface cung cấp dữ liệu thị trường tuần tự theo thời gian.
    Đảm bảo không bao giờ lộ dữ liệu tương lai (Anti-Lookahead Bias).
    """
    @abstractmethod
    def get_current_bar(self, symbol: str, timeframe: str | None = None) -> pd.Series | None:
        raise NotImplementedError

    @abstractmethod
    def get_history(self, symbol: str, count: int, timeframe: str | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def advance(self) -> bool:
        raise NotImplementedError


class SingleAssetDataFeed(DataFeed):
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        self._df = market_data.data.reset_index(drop=True)
        self._cursor: int = 0

    @property
    def current_time(self) -> pd.Timestamp:
        return pd.Timestamp(self._df.iloc[self._cursor]["time"])

    def reset(self):
        self._cursor = 0

    def get_current_bar(self, symbol: str, timeframe: str | None = None) -> pd.Series | None:
        if symbol.upper() != self.market_data.symbol:
            raise KeyError(f"Symbol '{symbol}' không có trong SingleAssetDataFeed ({self.market_data.symbol}).")
        if timeframe is not None and timeframe.upper() != self.market_data.timeframe:
            raise KeyError(f"Timeframe '{timeframe}' không khớp ({self.market_data.timeframe}).")
        return self._df.iloc[self._cursor]

    def get_history(self, symbol: str, count: int, timeframe: str | None = None) -> pd.DataFrame:
        if symbol.upper() != self.market_data.symbol:
            raise KeyError(f"Symbol '{symbol}' không có trong SingleAssetDataFeed ({self.market_data.symbol}).")
        if count <= 0:
            return self._df.iloc[0:0].copy()
        start_idx = max(0, self._cursor - count + 1)
        return self._df.iloc[start_idx : self._cursor + 1].copy().reset_index(drop=True)

    def advance(self) -> bool:
        if self._cursor + 1 < len(self._df):
            self._cursor += 1
            return True
        return False

