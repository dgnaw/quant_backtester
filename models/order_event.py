from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class OrderEvent:
    order_id: str
    event_type: str
    timestamp: pd.Timestamp
    bar_index: int
    reason: str = ""
    filled_quantity: float = 0.0
    fill_price: float | None = None

