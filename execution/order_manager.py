from __future__ import annotations
import pandas as pd
from quant_backtester.models.order import Order, OrderIntent, OrderSide, OrderType, OrderStatus


class OrderManager:
    def __init__(self):
        self.orders: list[Order] = []
        self._order_counter = 0

    def create_order(
        self,
        time: pd.Timestamp,
        symbol: str,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        quantity: float = 0.01,
        units: float = 1.0,
        limit_price: float | None = None,
        stop_price: float | None = None,
        reason: str = "INITIAL",
    ) -> Order:
        self._order_counter += 1
        order = Order(
            order_id=f"ORD-{self._order_counter:05d}",
            time=time,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(quantity),
            units=float(units),
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            reason=reason,
        )
        self.orders.append(order)
        return order

    def create_from_intent(self, intent: OrderIntent, time: pd.Timestamp, units: float) -> Order:
        qty = intent.quantity if intent.quantity is not None else 0.01
        return self.create_order(
            time=time,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=qty,
            units=units,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            reason=intent.reason,
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame(columns=["order_id", "time", "symbol", "side", "order_type", "quantity", "units", "status", "reason"])
        return pd.DataFrame([
            {
                "order_id": o.order_id,
                "time": o.time,
                "symbol": o.symbol,
                "side": o.side.name,
                "order_type": o.order_type.value,
                "quantity": o.quantity,
                "units": o.units,
                "status": o.status.value,
                "reason": o.reason,
            }
            for o in self.orders
        ])

