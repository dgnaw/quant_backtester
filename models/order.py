from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import math
from typing import Any
import pandas as pd


class OrderSide(Enum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP = "STOP_MARKET"  # Alias tương thích ngược


class OrderStatus(Enum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"
    DAY = "DAY"


class OrderOrigin(Enum):
    STRATEGY_ENTRY = "STRATEGY_ENTRY"
    STRATEGY_EXIT = "STRATEGY_EXIT"
    PROTECTIVE_SL = "PROTECTIVE_SL"
    PROTECTIVE_TP = "PROTECTIVE_TP"
    END_OF_TEST = "END_OF_TEST"


class ExecutionPhase(Enum):
    OPEN = "OPEN"
    RANGE = "RANGE"
    CLOSE = "CLOSE"


TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.REJECTED,
    OrderStatus.CANCELLED,
    OrderStatus.EXPIRED,
}


def validate_and_normalize_lot(
    quantity: float,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    lot_step: float = 0.01,
) -> float:
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
        raise ValueError("quantity phải là số thực.")
    q_float = float(quantity)
    if not math.isfinite(q_float) or q_float <= 0.0:
        raise ValueError("quantity phải hữu hạn và > 0.")

    try:
        d_qty = Decimal(f"{q_float:.8f}")
        d_min = Decimal(f"{float(min_lot):.8f}")
        d_max = Decimal(f"{float(max_lot):.8f}")
        d_step = Decimal(f"{float(lot_step):.8f}")
    except InvalidOperation as exc:
        raise ValueError(f"quantity không hợp lệ: {quantity}") from exc

    if d_qty < d_min:
        raise ValueError(f"quantity ({q_float}) nhỏ hơn min_lot ({min_lot}).")
    if d_qty > d_max:
        raise ValueError(f"quantity ({q_float}) lớn hơn max_lot ({max_lot}).")

    if d_step > 0:
        remainder = d_qty % d_step
        if remainder != Decimal("0") and abs(remainder - d_step) > Decimal("1e-7") and remainder > Decimal("1e-7"):
            raise ValueError(f"quantity ({q_float}) không tuân thủ bước khối lượng lot_step ({lot_step}).")
        steps = int((d_qty / d_step).to_integral_value())
        return float(steps * d_step)

    return float(d_qty)


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
    time_in_force: TimeInForce | None = None
    expire_after_bars: int | None = None
    origin: OrderOrigin = OrderOrigin.STRATEGY_ENTRY
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
    time_in_force: TimeInForce | None = None
    expire_after_bars: int | None = None
    created_bar_index: int = 0
    eligible_from_bar: int = 0
    filled_quantity: float = 0.0
    remaining_quantity: float | None = None
    origin: OrderOrigin = OrderOrigin.STRATEGY_ENTRY
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    reason: str = "INITIAL"
    reject_reason: str | None = None
    cancel_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, (int, float)) or not math.isfinite(float(self.quantity)) or float(self.quantity) <= 0.0:
            raise ValueError("Order.quantity phải là số dương hữu hạn (> 0).")
        self.quantity = float(self.quantity)
        self.units = float(self.units)

        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None or not math.isfinite(float(self.limit_price)) or float(self.limit_price) <= 0.0:
                raise ValueError("Order LIMIT bắt buộc phải có limit_price hợp lệ (> 0).")
            self.limit_price = float(self.limit_price)

        if self.order_type == OrderType.STOP_MARKET:
            if self.stop_price is None or not math.isfinite(float(self.stop_price)) or float(self.stop_price) <= 0.0:
                raise ValueError("Order STOP_MARKET bắt buộc phải có stop_price hợp lệ (> 0).")
            self.stop_price = float(self.stop_price)

        if self.time_in_force is None:
            if self.order_type == OrderType.MARKET:
                self.time_in_force = TimeInForce.IOC
            else:
                self.time_in_force = TimeInForce.GTC

        if self.expire_after_bars is not None:
            if not isinstance(self.expire_after_bars, int) or isinstance(self.expire_after_bars, bool) or self.expire_after_bars <= 0:
                raise ValueError("expire_after_bars phải là số nguyên dương (> 0).")

        self.filled_quantity = float(self.filled_quantity)
        if self.remaining_quantity is None:
            self.remaining_quantity = max(0.0, self.quantity - self.filled_quantity)
        else:
            self.remaining_quantity = float(self.remaining_quantity)

    @property
    def created_time(self) -> pd.Timestamp:
        return self.time

    @property
    def price(self) -> float | None:
        return self.limit_price if self.limit_price is not None else self.stop_price

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_ORDER_STATUSES
