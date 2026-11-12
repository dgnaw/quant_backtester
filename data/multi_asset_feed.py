from __future__ import annotations
import pandas as pd

from .data_feed import DataFeed
from .market_data import MarketData
from .time_utils import parse_timeframe_delta


class MultiAssetDataFeed(DataFeed):
    """
    Quản lý và đồng bộ dữ liệu của nhiều Symbol và nhiều Timeframe (M15, H1, D1...).
    Tuân thủ nghiêm ngặt quy tắc Chống Nhìn Trộm Tương Lai (Anti-Lookahead Bias):
    - Một cây nến ở khung thời gian lớn (ví dụ H1 mở lúc 10:00, đóng lúc 11:00)
      CHỈ được phép truy cập khi mốc thời gian hoàn tất của bước hiện tại >= 11:00.
    """
    def __init__(self):
        self._datasets: dict[tuple[str, str], MarketData] = {}
        self._default_tf_per_symbol: dict[str, str] = {}
        self._base_delta: pd.Timedelta | None = None
        self._timeline: list[pd.Timestamp] = []
        self._cursor: int = 0

    def add_market_data(self, market_data: MarketData):
        key = (market_data.symbol, market_data.timeframe)
        self._datasets[key] = market_data

        tf_delta = parse_timeframe_delta(market_data.timeframe)
        if market_data.symbol not in self._default_tf_per_symbol:
            self._default_tf_per_symbol[market_data.symbol] = market_data.timeframe
        else:
            existing_delta = parse_timeframe_delta(self._default_tf_per_symbol[market_data.symbol])
            if tf_delta < existing_delta:
                self._default_tf_per_symbol[market_data.symbol] = market_data.timeframe

        if self._base_delta is None or tf_delta < self._base_delta:
            self._base_delta = tf_delta

        self._rebuild_timeline()

    def _rebuild_timeline(self):
        if not self._datasets or self._base_delta is None:
            self._timeline = []
            return

        # Timeline chính được xây dựng từ các dataset có khung thời gian nhỏ nhất (base timeframe)
        base_times: set[pd.Timestamp] = set()
        for (sym, tf), md in self._datasets.items():
            if parse_timeframe_delta(tf) == self._base_delta:
                base_times.update(pd.to_datetime(md.data["time"]).tolist())

        self._timeline = sorted(base_times)
        self._cursor = 0

    @property
    def current_time(self) -> pd.Timestamp:
        if not self._timeline:
            raise RuntimeError("MultiAssetDataFeed chưa có dữ liệu.")
        return self._timeline[self._cursor]

    @property
    def current_evaluation_close_time(self) -> pd.Timestamp:
        """Thời điểm đóng cửa của cây nến khung cơ sở hiện tại."""
        assert self._base_delta is not None
        return self.current_time + self._base_delta

    def _resolve_key(self, symbol: str, timeframe: str | None) -> tuple[str, str]:
        sym = symbol.strip().upper()
        if timeframe is None:
            if sym not in self._default_tf_per_symbol:
                raise KeyError(f"Symbol '{sym}' không tồn tại trong MultiAssetDataFeed.")
            tf = self._default_tf_per_symbol[sym]
        else:
            tf = timeframe.strip().upper()

        key = (sym, tf)
        if key not in self._datasets:
            raise KeyError(f"Không tìm thấy dữ liệu cho {key}.")
        return key

    def get_current_bar(self, symbol: str, timeframe: str | None = None) -> pd.Series | None:
        hist = self.get_history(symbol, count=1, timeframe=timeframe)
        if hist.empty:
            return None
        return hist.iloc[-1]

    def get_history(self, symbol: str, count: int, timeframe: str | None = None) -> pd.DataFrame:
        key = self._resolve_key(symbol, timeframe)
        md = self._datasets[key]
        tf_delta = parse_timeframe_delta(md.timeframe)
        eval_close_time = self.current_evaluation_close_time

        df = md.data
        # Chỉ những cây nến đã ĐÓNG CỬA hoàn toàn tại thời điểm eval_close_time mới được phép nhìn thấy:
        # bar_open_time + tf_delta <= eval_close_time
        completed_mask = (df["time"] + tf_delta) <= eval_close_time
        visible_df = df.loc[completed_mask]

        if count <= 0 or visible_df.empty:
            return df.iloc[0:0].copy()

        return visible_df.iloc[-count:].copy().reset_index(drop=True)

    def advance(self) -> bool:
        if self._cursor + 1 < len(self._timeline):
            self._cursor += 1
            return True
        return False

