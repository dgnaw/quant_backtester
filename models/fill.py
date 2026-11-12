from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from .order import OrderSide


@dataclass
class Fill:
    fill_id: str
    order_id: str
    time: pd.Timestamp
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    units: float
    fee: float = 0.0
    slippage: float = 0.0
    reason: str = "EXECUTION"
    metadata: dict[str, Any] = field(default_factory=dict)

