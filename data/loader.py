from __future__ import annotations
from pathlib import Path
import sqlite3
import pandas as pd

from .schema import normalize_schema
from .validator import DataValidator, DataQualityPolicy
from .time_utils import TimeConfig, normalize_timestamps
from .market_data import MarketData


class DataLoader:
    """
    Đọc dữ liệu từ các nguồn lưu trữ (CSV, SQLite), chuẩn hóa Schema,
    chuyển đổi Timezone về UTC, kiểm tra chất lượng qua DataValidator và trả về MarketData.
    """
    def __init__(
        self,
        validator: DataValidator | None = None,
        time_config: TimeConfig | None = None,
    ):
        self.validator = validator if validator is not None else DataValidator(DataQualityPolicy())
        self.time_config = time_config if time_config is not None else TimeConfig()

    def _prepare_market_data(
        self,
        raw_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str,
        price_type: str = "MID",
    ) -> MarketData:
        norm_df, vol_type = normalize_schema(raw_df)
        norm_df["time"] = normalize_timestamps(norm_df["time"], self.time_config)

        report = self.validator.validate(norm_df)
        if not report.is_valid:
            err_msgs = "; ".join(f"[{e.code}] {e.message}" for e in report.errors)
            raise ValueError(f"Dữ liệu không hợp lệ ({symbol} {timeframe}): {err_msgs}")

        for col in ("open", "high", "low", "close", "volume"):
            norm_df[col] = pd.to_numeric(norm_df[col], errors="raise").astype(float)

        return MarketData(
            data=norm_df.reset_index(drop=True),
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            price_type=price_type,
            timezone=self.time_config.target_timezone,
            volume_type=vol_type,
            extra_metadata={"validation_report": report},
        )

    def load_csv(
        self,
        path: str | Path,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
        price_type: str = "MID",
    ) -> MarketData:
        file_path = Path(path)
        raw_df = pd.read_csv(file_path)
        return self._prepare_market_data(
            raw_df=raw_df,
            symbol=symbol,
            timeframe=timeframe,
            source=f"CSV:{file_path.name}",
            price_type=price_type,
        )

    def load_sqlite(
        self,
        db_path: str | Path,
        table_name: str,
        symbol: str | None = None,
        timeframe: str | None = None,
        price_type: str = "MID",
    ) -> MarketData:
        resolved_symbol = symbol
        resolved_tf = timeframe
        if (resolved_symbol is None or resolved_tf is None) and "_" in table_name:
            parts = table_name.split("_", 1)
            resolved_symbol = resolved_symbol or parts[0]
            resolved_tf = resolved_tf or parts[1]

        resolved_symbol = resolved_symbol or table_name
        resolved_tf = resolved_tf or "M15"

        with sqlite3.connect(str(db_path)) as conn:
            raw_df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY time ASC", conn)

        return self._prepare_market_data(
            raw_df=raw_df,
            symbol=resolved_symbol,
            timeframe=resolved_tf,
            source=f"SQLITE:{Path(db_path).name}:{table_name}",
            price_type=price_type,
        )

