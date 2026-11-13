from __future__ import annotations
from typing import Any
import pandas as pd

from quant_backtester.data.trading_calendar import TradingCalendar
from quant_backtester.models.order import (
    TERMINAL_ORDER_STATUSES,
    Order,
    OrderIntent,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_backtester.models.order_event import OrderEvent


class OrderManager:
    def __init__(self, calendar: TradingCalendar | None = None):
        self.orders: list[Order] = []
        self.events: list[OrderEvent] = []
        self._order_counter = 0
        self.calendar = calendar if calendar is not None else TradingCalendar()
        self._pending_entry_by_symbol: dict[str, Order] = {}

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD-{self._order_counter:05d}"

    def _record_event(
        self,
        order_id: str,
        event_type: str,
        timestamp: pd.Timestamp,
        bar_index: int,
        reason: str = "",
        filled_quantity: float = 0.0,
        fill_price: float | None = None,
    ) -> None:
        self.events.append(
            OrderEvent(
                order_id=order_id,
                event_type=event_type,
                timestamp=pd.Timestamp(timestamp),
                bar_index=int(bar_index),
                reason=reason,
                filled_quantity=float(filled_quantity),
                fill_price=fill_price,
            )
        )

    def get_pending_entry_order(self, symbol: str) -> Order | None:
        existing = self._pending_entry_by_symbol.get(symbol)
        if existing is not None and existing.status == OrderStatus.PENDING:
            return existing
        return None

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
        bar_index: int = 0,
        eligible_from_bar: int | None = None,
        time_in_force: TimeInForce | None = None,
        expire_after_bars: int | None = None,
        origin: OrderOrigin = OrderOrigin.STRATEGY_ENTRY,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Order:
        # D7 Pending Order Policy: Nếu tạo Entry Order mới hợp lệ, hủy Pending Entry Order cũ cùng symbol với reason SUPERSEDED
        if origin == OrderOrigin.STRATEGY_ENTRY:
            old_pending = self.get_pending_entry_order(symbol)
            if old_pending is not None:
                self.cancel_order(old_pending, timestamp=time, bar_index=bar_index, reason="SUPERSEDED")

        oid = self._next_order_id()
        eff_eligible = eligible_from_bar if eligible_from_bar is not None else bar_index

        order = Order(
            order_id=oid,
            time=pd.Timestamp(time),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(quantity),
            units=float(units),
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.CREATED,
            time_in_force=time_in_force,
            expire_after_bars=expire_after_bars,
            created_bar_index=int(bar_index),
            eligible_from_bar=int(eff_eligible),
            origin=origin,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            reason=reason,
            metadata=dict(metadata) if metadata else {},
        )
        self._record_event(oid, "CREATED", time, bar_index, reason=reason)

        # Chuyển CREATED -> PENDING
        order.status = OrderStatus.PENDING
        self._record_event(oid, "PENDING", time, bar_index, reason=reason)
        self.orders.append(order)

        if origin == OrderOrigin.STRATEGY_ENTRY:
            self._pending_entry_by_symbol[symbol] = order

        return order

    def create_rejected_order(
        self,
        time: pd.Timestamp,
        symbol: str,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        quantity: float = 0.01,
        units: float = 1.0,
        limit_price: float | None = None,
        stop_price: float | None = None,
        bar_index: int = 0,
        reject_reason: str = "RISK_REJECTED",
        origin: OrderOrigin = OrderOrigin.STRATEGY_ENTRY,
    ) -> Order:
        oid = self._next_order_id()
        safe_qty = float(quantity) if quantity and float(quantity) > 0 else 0.01
        safe_units = float(units) if units and float(units) > 0 else 1.0
        order = Order(
            order_id=oid,
            time=pd.Timestamp(time),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=safe_qty,
            units=safe_units,
            limit_price=limit_price if order_type != OrderType.LIMIT or limit_price else 1.0,
            stop_price=stop_price if order_type != OrderType.STOP_MARKET or stop_price else 1.0,
            status=OrderStatus.CREATED,
            created_bar_index=int(bar_index),
            eligible_from_bar=int(bar_index),
            origin=origin,
            reason="REJECTED_AT_CREATION",
            reject_reason=reject_reason,
        )
        self._record_event(oid, "CREATED", time, bar_index, reason="CREATED")
        order.status = OrderStatus.REJECTED
        self._record_event(oid, "REJECTED", time, bar_index, reason=reject_reason)
        self.orders.append(order)
        return order

    def create_from_intent(
        self,
        intent: OrderIntent,
        time: pd.Timestamp,
        units: float,
        bar_index: int = 0,
        eligible_from_bar: int | None = None,
    ) -> Order:
        qty = intent.quantity if intent.quantity is not None else 0.01
        origin = intent.origin
        if intent.reason == "STRATEGY_EXIT":
            origin = OrderOrigin.STRATEGY_EXIT
        elif intent.reason in ("STOP_LOSS", "STOP_GAP"):
            origin = OrderOrigin.PROTECTIVE_SL
        elif intent.reason in ("TAKE_PROFIT", "TAKE_PROFIT_GAP"):
            origin = OrderOrigin.PROTECTIVE_TP
        elif intent.reason == "END_OF_TEST":
            origin = OrderOrigin.END_OF_TEST

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
            bar_index=bar_index,
            eligible_from_bar=eligible_from_bar,
            time_in_force=intent.time_in_force,
            expire_after_bars=intent.expire_after_bars,
            origin=origin,
            stop_loss_price=intent.stop_loss_price,
            take_profit_price=intent.take_profit_price,
            stop_loss_pct=intent.stop_loss_pct,
            take_profit_pct=intent.take_profit_pct,
            metadata=intent.metadata,
        )

    def mark_filled(
        self,
        order: Order,
        fill_price: float,
        filled_quantity: float | None = None,
        timestamp: pd.Timestamp | None = None,
        bar_index: int = 0,
        reason: str = "FILLED",
    ) -> None:
        if order.status in TERMINAL_ORDER_STATUSES:
            raise ValueError(f"Không thể Fill Order '{order.order_id}' đã ở trạng thái kết thúc ({order.status.value}).")
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Chỉ Order đang PENDING mới được chuyển sang FILLED (hiện tại: {order.status.value}).")

        fq = float(filled_quantity) if filled_quantity is not None else order.quantity
        order.filled_quantity = fq
        order.remaining_quantity = max(0.0, order.quantity - fq)
        order.status = OrderStatus.FILLED

        ts = timestamp if timestamp is not None else order.time
        self._record_event(order.order_id, "FILLED", ts, bar_index, reason=reason, filled_quantity=fq, fill_price=float(fill_price))

        if self._pending_entry_by_symbol.get(order.symbol) is order:
            self._pending_entry_by_symbol.pop(order.symbol, None)

    def reject_pending_order(
        self,
        order: Order,
        timestamp: pd.Timestamp,
        bar_index: int = 0,
        reason: str = "POST_FILL_VALIDATION_FAILED",
    ) -> None:
        if order.status in TERMINAL_ORDER_STATUSES:
            raise ValueError(f"Order '{order.order_id}' đã ở trạng thái kết thúc ({order.status.value}).")
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        self._record_event(order.order_id, "REJECTED", timestamp, bar_index, reason=reason)
        if self._pending_entry_by_symbol.get(order.symbol) is order:
            self._pending_entry_by_symbol.pop(order.symbol, None)

    def cancel_order(
        self,
        order: Order,
        timestamp: pd.Timestamp,
        bar_index: int = 0,
        reason: str = "USER_CANCELLED",
    ) -> None:
        if order.status in TERMINAL_ORDER_STATUSES:
            raise ValueError(f"Không thể hủy Order '{order.order_id}' đã ở trạng thái kết thúc ({order.status.value}).")
        order.status = OrderStatus.CANCELLED
        order.cancel_reason = reason
        self._record_event(order.order_id, "CANCELLED", timestamp, bar_index, reason=reason)
        if self._pending_entry_by_symbol.get(order.symbol) is order:
            self._pending_entry_by_symbol.pop(order.symbol, None)

    def expire_order(
        self,
        order: Order,
        timestamp: pd.Timestamp,
        bar_index: int = 0,
        reason: str = "TIME_IN_FORCE_EXPIRED",
    ) -> None:
        if order.status in TERMINAL_ORDER_STATUSES:
            raise ValueError(f"Không thể Expire Order '{order.order_id}' đã ở trạng thái kết thúc ({order.status.value}).")
        order.status = OrderStatus.EXPIRED
        order.cancel_reason = reason
        self._record_event(order.order_id, "EXPIRED", timestamp, bar_index, reason=reason)
        if self._pending_entry_by_symbol.get(order.symbol) is order:
            self._pending_entry_by_symbol.pop(order.symbol, None)

    def check_expirations_after_bar(
        self,
        bar_index: int,
        timestamp: pd.Timestamp,
        next_timestamp: pd.Timestamp | None = None,
    ) -> None:
        """
        Kiểm tra hiệu lực TimeInForce sau khi kết thúc xử lý bar_index:
        - IOC: Nếu đã đủ điều kiện tại bar_index (bar_index >= eligible_from_bar) mà vẫn PENDING -> CANCELLED (IOC_UNFILLED).
        - GTC có expire_after_bars = K: Có hiệu lực tại các bar [eligible_from_bar .. eligible_from_bar + K - 1].
          Sau khi kết thúc bar (eligible_from_bar + K - 1) mà vẫn PENDING -> EXPIRED (TIME_IN_FORCE_EXPIRED).
        - DAY: Nếu bar tiếp theo thuộc phiên khác (session_id khác) hoặc kết thúc phiên -> EXPIRED (DAY_SESSION_EXPIRED).
        """
        for order in list(self.orders):
            if order.status != OrderStatus.PENDING:
                continue
            if bar_index < order.eligible_from_bar:
                continue

            if order.time_in_force == TimeInForce.IOC:
                self.cancel_order(order, timestamp=timestamp, bar_index=bar_index, reason="IOC_UNFILLED")
                continue

            if order.expire_after_bars is not None:
                last_valid_bar = order.eligible_from_bar + order.expire_after_bars - 1
                if bar_index >= last_valid_bar:
                    self.expire_order(order, timestamp=timestamp, bar_index=bar_index, reason="TIME_IN_FORCE_EXPIRED")
                    continue

            if order.time_in_force == TimeInForce.DAY and next_timestamp is not None:
                curr_sess = self.calendar.session_id(timestamp)
                next_sess = self.calendar.session_id(next_timestamp)
                if curr_sess != next_sess:
                    self.expire_order(order, timestamp=timestamp, bar_index=bar_index, reason="DAY_SESSION_EXPIRED")

    def expire_all_pending(
        self,
        timestamp: pd.Timestamp,
        bar_index: int,
        reason: str = "END_OF_TEST",
    ) -> None:
        for order in list(self.orders):
            if order.status == OrderStatus.PENDING:
                self.expire_order(order, timestamp=timestamp, bar_index=bar_index, reason=reason)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame(
                columns=[
                    "order_id",
                    "time",
                    "symbol",
                    "side",
                    "order_type",
                    "quantity",
                    "filled_quantity",
                    "remaining_quantity",
                    "units",
                    "status",
                    "time_in_force",
                    "origin",
                    "reason",
                    "reject_reason",
                    "cancel_reason",
                ]
            )
        return pd.DataFrame([
            {
                "order_id": o.order_id,
                "time": o.time,
                "symbol": o.symbol,
                "side": o.side.name,
                "order_type": o.order_type.value,
                "quantity": o.quantity,
                "filled_quantity": o.filled_quantity,
                "remaining_quantity": o.remaining_quantity,
                "units": o.units,
                "status": o.status.value,
                "time_in_force": o.time_in_force.value if o.time_in_force else None,
                "origin": o.origin.value,
                "reason": o.reason,
                "reject_reason": o.reject_reason,
                "cancel_reason": o.cancel_reason,
            }
            for o in self.orders
        ])

    def events_to_dataframe(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=["order_id", "event_type", "timestamp", "bar_index", "reason", "filled_quantity", "fill_price"]
            )
        return pd.DataFrame([
            {
                "order_id": e.order_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "bar_index": e.bar_index,
                "reason": e.reason,
                "filled_quantity": e.filled_quantity,
                "fill_price": e.fill_price,
            }
            for e in self.events
        ])
