from __future__ import annotations
import math
from typing import Any
import pandas as pd

from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import StrategySignals

REQUIRED_SIGNAL_COLUMNS = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "entry_signal",
    "exit_long_signal",
    "exit_short_signal",
}

DIRECTIVE_COLUMNS = (
    "quantity",
    "limit_price",
    "stop_price",
    "stop_loss_price",
    "take_profit_price",
    "stop_loss_pct",
    "take_profit_pct",
    "trailing_stop_pct",
    "expire_after_bars",
)


def _is_valid_number(val: Any) -> bool:
    if val is None or pd.isna(val):
        return False
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return math.isfinite(float(val))
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def frame_to_signals(data: StrategySignals | pd.DataFrame, symbol: str) -> list[Signal]:
    """
    Chuyển đổi StrategySignals hoặc DataFrame legacy 3 cột sang danh sách Signal chuẩn.
    Quy tắc ưu tiên khi nhiều cột cùng kích hoạt trên 1 nến:
    1. Nếu cả exit_long_signal và exit_short_signal đều True -> EXIT_ALL (nếu entry == 0) hoặc ưu tiên Exit trước Entry.
    2. Exit (EXIT_LONG / EXIT_SHORT / EXIT_ALL) được ưu tiên hơn Entry nếu chỉ có Exit,
       nhưng để tương thích chính xác với legacy Engine:
       Nếu có cả Entry và Exit trên cùng 1 nến (ví dụ tùy trạng thái vị thế đang FLAT hay đang giữ lệnh),
       ta lưu kèm thông tin raw flags trong metadata["legacy_flags"] để SignalTranslator hoặc Engine xử lý chính xác
       khi vị thế hiện tại là FLAT (xét entry) hay đang mở vị thế (xét exit).
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol không được rỗng.")

    df = data.data if isinstance(data, StrategySignals) else data
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Đầu vào của frame_to_signals phải là DataFrame hoặc StrategySignals.")

    missing = REQUIRED_SIGNAL_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong Strategy output: {sorted(missing)}")

    signals: list[Signal] = []
    present_directive_cols = [c for c in DIRECTIVE_COLUMNS if c in df.columns]

    for idx, row in df.iterrows():
        raw_time = row["time"]
        if pd.isna(raw_time):
            raise ValueError(f"Timestamp không hợp lệ (NaT/None) tại dòng {idx}.")
        ts = pd.Timestamp(raw_time)

        raw_entry = row["entry_signal"]
        if isinstance(raw_entry, bool) or (not pd.isna(raw_entry) and not isinstance(raw_entry, (int, float)) and not hasattr(raw_entry, "__int__")):
            try:
                if isinstance(raw_entry, str):
                    raise ValueError(f"entry_signal không hợp lệ tại dòng {idx}: {raw_entry!r}")
                entry_val = int(raw_entry)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"entry_signal không hợp lệ tại dòng {idx}: {raw_entry!r}") from exc
        else:
            entry_val = 0 if pd.isna(raw_entry) else int(raw_entry)

        if entry_val not in (-1, 0, 1):
            entry_val = 1 if entry_val > 0 else (-1 if entry_val < 0 else 0)

        raw_exit_long = row["exit_long_signal"]
        raw_exit_short = row["exit_short_signal"]
        exit_long = False if pd.isna(raw_exit_long) else bool(raw_exit_long)
        exit_short = False if pd.isna(raw_exit_short) else bool(raw_exit_short)

        # Xác định SignalType chính theo quy tắc rõ ràng:
        # Ưu tiên: Entry nếu có entry_val != 0; nếu entry_val == 0 thì xét Exit.
        # Đồng thời nếu cả Entry và Exit cùng xuất hiện trên 1 nến, ghi nhận vào legacy_flags
        # để Translator đối chiếu với vị thế thực tế (khi FLAT thì dùng Entry, khi LONG/SHORT thì dùng Exit).
        if entry_val > 0:
            sig_type = SignalType.LONG
        elif entry_val < 0:
            sig_type = SignalType.SHORT
        elif exit_long and exit_short:
            sig_type = SignalType.EXIT_ALL
        elif exit_long:
            sig_type = SignalType.EXIT_LONG
        elif exit_short:
            sig_type = SignalType.EXIT_SHORT
        else:
            sig_type = SignalType.FLAT

        directives: dict[str, Any] = {}
        for col in present_directive_cols:
            val = row[col]
            if _is_valid_number(val):
                directives[col] = float(val)

        meta: dict[str, Any] = {
            "directives": directives,
            "legacy_flags": {
                "entry_signal": entry_val,
                "exit_long_signal": exit_long,
                "exit_short_signal": exit_short,
            },
        }

        signals.append(
            Signal(
                time=ts,
                symbol=symbol,
                signal_type=sig_type,
                strength=1.0,
                metadata=meta,
            )
        )

    return signals

