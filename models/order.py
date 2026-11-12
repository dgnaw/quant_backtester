from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import pandas as pd


class OrderSide(Enum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    reason: str = "STRATEGY_SIGNAL"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    order_id: str
    time: pd.Timestamp
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    units: float
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    reason: str = "INITIAL"
    reject_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

