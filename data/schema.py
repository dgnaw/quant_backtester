from __future__ import annotations
import pandas as pd

STANDARD_COLUMNS: tuple[str, ...] = ("time", "open", "high", "low", "close", "volume")
REQUIRED_OHLC_COLUMNS: tuple[str, ...] = ("time", "open", "high", "low", "close")
REQUIRED_COLUMNS: set[str] = set(REQUIRED_OHLC_COLUMNS)
OPTIONAL_QUOTE_COLUMNS: tuple[str, ...] = (
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "spread",
)


def normalize_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Chuẩn hóa DataFrame đầu vào về Schema OHLCV tiêu chuẩn:
    (time, open, high, low, close, volume) kèm các cột Bid/Ask tùy chọn nếu có,
    và trả về (df_normalized, volume_type).
    """
    out = df.copy()
    # Chuẩn hóa tên cột về chữ thường, bỏ khoảng trắng thừa
    col_map = {c: str(c).strip().lower() for c in out.columns}
    out = out.rename(columns=col_map)

    volume_type = "none"
    if "volume" in out.columns:
        volume_type = "volume"
    elif "real_volume" in out.columns:
        out["volume"] = out["real_volume"]
        volume_type = "real_volume"
    elif "tick_volume" in out.columns:
        out["volume"] = out["tick_volume"]
        volume_type = "tick_volume"
    else:
        out["volume"] = 0.0
        volume_type = "missing"

    missing = [c for c in REQUIRED_OHLC_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc trong Schema: {missing}")

    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    keep_cols = list(STANDARD_COLUMNS) + [c for c in OPTIONAL_QUOTE_COLUMNS if c in out.columns]
    return out[keep_cols].copy(), volume_type
