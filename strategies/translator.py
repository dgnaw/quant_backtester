from __future__ import annotations
from typing import Any, Mapping

from quant_backtester.config import PositionMode, ReversePolicy
from quant_backtester.models.order import OrderIntent, OrderOrigin, OrderSide, OrderType, TimeInForce
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.portfolio.position import PositionSide, PositionView


class SignalTranslator:
    """
    Chuyển đổi Signal + PositionView chỉ đọc thành OrderIntent | None.
    Thực thi bảng quyết định 15 trạng thái (Position × Signal) và ghi lại signal_log.
    Hỗ trợ PositionMode (NETTING_SINGLE mặc định, NETTING_SCALE_IN) và Partial Close qua directives.
    """

    def __init__(
        self,
        default_quantity: float | None = 0.01,
        reverse_policy: ReversePolicy = ReversePolicy.IGNORE,
        position_mode: PositionMode = PositionMode.NETTING_SINGLE,
        position_mode_by_symbol: dict[str, PositionMode] | None = None,
    ):
        self.default_quantity = default_quantity
        self.reverse_policy = reverse_policy
        self.position_mode = position_mode
        self.position_mode_by_symbol: dict[str, PositionMode] = (
            dict(position_mode_by_symbol) if position_mode_by_symbol else {}
        )
        self.signal_log: list[dict[str, Any]] = []

    def get_position_mode(self, symbol: str) -> PositionMode:
        return self.position_mode_by_symbol.get(symbol, self.position_mode)

    def _effective_signal_type(self, signal: Signal, position_view: PositionView) -> SignalType:
        """
        Hỗ trợ tương thích ngược cho DataFrame legacy khi một dòng vừa có entry_signal vừa có exit_*_signal
        (ví dụ: nến hiện tại thoả điều kiện exit_long_signal=True khi đang giữ LONG, dù entry_signal=0 hoặc != 0).
        """
        meta = signal.metadata or {}
        flags = meta.get("legacy_flags")
        if not flags:
            return signal.signal_type

        entry_val = flags.get("entry_signal", 0)
        exit_long = flags.get("exit_long_signal", False)
        exit_short = flags.get("exit_short_signal", False)

        if position_view.is_flat:
            if entry_val > 0:
                return SignalType.LONG
            if entry_val < 0:
                return SignalType.SHORT
            if exit_long and exit_short:
                return SignalType.EXIT_ALL
            if exit_long:
                return SignalType.EXIT_LONG
            if exit_short:
                return SignalType.EXIT_SHORT
            return SignalType.FLAT

        if position_view.is_long:
            if exit_long:
                return SignalType.EXIT_LONG
            if entry_val < 0:
                return SignalType.SHORT
            if entry_val > 0:
                return SignalType.LONG
            if exit_short:
                return SignalType.EXIT_SHORT
            return SignalType.FLAT

        if position_view.is_short:
            if exit_short:
                return SignalType.EXIT_SHORT
            if entry_val > 0:
                return SignalType.LONG
            if entry_val < 0:
                return SignalType.SHORT
            if exit_long:
                return SignalType.EXIT_LONG
            return SignalType.FLAT

        return signal.signal_type

    def _build_entry_intent(self, signal: Signal, side: OrderSide, reason: str) -> OrderIntent:
        meta = signal.metadata or {}
        raw_directives = meta.get("directives", {}) if isinstance(meta, Mapping) else {}
        directives = dict(raw_directives) if isinstance(raw_directives, Mapping) else {}

        qty = directives.get("quantity", self.default_quantity)
        limit_price = directives.get("limit_price")
        stop_price = directives.get("stop_price")
        sl_price = directives.get("stop_loss_price")
        tp_price = directives.get("take_profit_price")
        sl_pct = directives.get("stop_loss_pct")
        tp_pct = directives.get("take_profit_pct")
        trailing_pct = directives.get("trailing_stop_pct")
        expire_bars = directives.get("expire_after_bars")
        if expire_bars is not None:
            expire_bars = int(expire_bars)

        raw_otype = directives.get("order_type", OrderType.MARKET)
        if isinstance(raw_otype, OrderType):
            otype = raw_otype
        elif isinstance(raw_otype, str):
            otype = OrderType[raw_otype.strip().upper()]
        else:
            otype = OrderType.MARKET

        raw_tif = directives.get("time_in_force")
        if isinstance(raw_tif, TimeInForce):
            tif = raw_tif
        elif isinstance(raw_tif, str):
            tif = TimeInForce[raw_tif.strip().upper()]
        else:
            tif = None

        return OrderIntent(
            symbol=signal.symbol,
            side=side,
            order_type=otype,
            quantity=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            trailing_stop_pct=trailing_pct,
            time_in_force=tif,
            expire_after_bars=expire_bars,
            origin=OrderOrigin.STRATEGY_ENTRY,
            reason=reason,
            metadata={"signal_time": signal.time, "directives": directives},
        )

    def _build_exit_intent(
        self,
        signal: Signal,
        close_side: OrderSide,
        reason: str = "STRATEGY_EXIT",
        extra_meta: dict[str, Any] | None = None,
    ) -> OrderIntent:
        sig_meta = signal.metadata or {}
        raw_directives = sig_meta.get("directives", {}) if isinstance(sig_meta, Mapping) else {}
        directives = dict(raw_directives) if isinstance(raw_directives, Mapping) else {}

        meta: dict[str, Any] = {"signal_time": signal.time}
        if "quantity" in directives and directives["quantity"] is not None:
            qty = float(directives["quantity"])
            meta["explicit_exit_quantity"] = True
            meta["directives"] = directives
        else:
            qty = self.default_quantity
            meta["explicit_exit_quantity"] = False

        if extra_meta:
            meta.update(extra_meta)
        return OrderIntent(
            symbol=signal.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=qty,
            origin=OrderOrigin.STRATEGY_EXIT,
            reason=reason,
            metadata=meta,
        )

    def _record_log(
        self,
        signal: Signal,
        sig_type: SignalType,
        position_before: PositionSide,
        decision: str,
        reason: str,
        intent: OrderIntent | None,
    ) -> None:
        self.signal_log.append(
            {
                "timestamp": signal.time,
                "symbol": signal.symbol,
                "signal_type": sig_type.name,
                "position_before": position_before.name,
                "decision": decision,
                "reason": reason,
                "intent_created": intent is not None,
            }
        )

    def translate(self, signal: Signal, position_view: PositionView) -> OrderIntent | None:
        if not isinstance(signal, Signal):
            raise TypeError("signal phải là một thực thể của Signal.")
        if signal.symbol != position_view.symbol:
            raise ValueError(
                f"Signal.symbol ({signal.symbol}) không khớp với PositionView.symbol ({position_view.symbol})."
            )

        sig_type = self._effective_signal_type(signal, position_view)
        pos_side = position_view.side if not position_view.is_flat else PositionSide.FLAT
        pos_mode = self.get_position_mode(signal.symbol)

        intent: OrderIntent | None = None
        decision = "NO_ACTION"
        reason = "NONE"

        # --- TRƯỜNG HỢP 1: POSITION = FLAT ---
        if pos_side == PositionSide.FLAT:
            if sig_type == SignalType.LONG:
                intent = self._build_entry_intent(signal, OrderSide.BUY, "ENTRY_LONG")
                decision = "OPEN_LONG"
                reason = "ENTRY_LONG"
            elif sig_type == SignalType.SHORT:
                intent = self._build_entry_intent(signal, OrderSide.SELL, "ENTRY_SHORT")
                decision = "OPEN_SHORT"
                reason = "ENTRY_SHORT"
            elif sig_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.EXIT_ALL):
                decision = "IGNORED"
                reason = "ALREADY_FLAT"
            else:
                decision = "NO_ACTION"
                reason = "SIGNAL_FLAT"

        # --- TRƯỜNG HỢP 2: POSITION = LONG ---
        elif pos_side == PositionSide.LONG:
            if sig_type == SignalType.LONG:
                if pos_mode == PositionMode.NETTING_SCALE_IN:
                    intent = self._build_entry_intent(signal, OrderSide.BUY, "SCALE_IN_LONG")
                    decision = "SCALE_IN_LONG"
                    reason = "SCALE_IN_LONG"
                else:
                    decision = "IGNORED"
                    reason = "ALREADY_LONG"
            elif sig_type in (SignalType.EXIT_LONG, SignalType.EXIT_ALL):
                intent = self._build_exit_intent(signal, OrderSide.SELL, "STRATEGY_EXIT")
                decision = "CLOSE_LONG"
                reason = "STRATEGY_EXIT"
            elif sig_type == SignalType.EXIT_SHORT:
                decision = "IGNORED"
                reason = "POSITION_IS_LONG"
            elif sig_type == SignalType.SHORT:
                if self.reverse_policy == ReversePolicy.IGNORE:
                    decision = "IGNORED"
                    reason = "REVERSE_POLICY_IGNORE"
                else:
                    intent = self._build_exit_intent(
                        signal,
                        OrderSide.SELL,
                        "STRATEGY_EXIT",
                        extra_meta={"reverse_to_side": OrderSide.SELL, "reverse_signal": signal, "explicit_exit_quantity": False},
                    )
                    decision = "REVERSE_CLOSE"
                    reason = "REVERSE_POLICY_REVERSE"
            else:
                decision = "NO_ACTION"
                reason = "SIGNAL_FLAT"

        # --- TRƯỜNG HỢP 3: POSITION = SHORT ---
        elif pos_side == PositionSide.SHORT:
            if sig_type == SignalType.SHORT:
                if pos_mode == PositionMode.NETTING_SCALE_IN:
                    intent = self._build_entry_intent(signal, OrderSide.SELL, "SCALE_IN_SHORT")
                    decision = "SCALE_IN_SHORT"
                    reason = "SCALE_IN_SHORT"
                else:
                    decision = "IGNORED"
                    reason = "ALREADY_SHORT"
            elif sig_type in (SignalType.EXIT_SHORT, SignalType.EXIT_ALL):
                intent = self._build_exit_intent(signal, OrderSide.BUY, "STRATEGY_EXIT")
                decision = "CLOSE_SHORT"
                reason = "STRATEGY_EXIT"
            elif sig_type == SignalType.EXIT_LONG:
                decision = "IGNORED"
                reason = "POSITION_IS_SHORT"
            elif sig_type == SignalType.LONG:
                if self.reverse_policy == ReversePolicy.IGNORE:
                    decision = "IGNORED"
                    reason = "REVERSE_POLICY_IGNORE"
                else:
                    intent = self._build_exit_intent(
                        signal,
                        OrderSide.BUY,
                        "STRATEGY_EXIT",
                        extra_meta={"reverse_to_side": OrderSide.BUY, "reverse_signal": signal, "explicit_exit_quantity": False},
                    )
                    decision = "REVERSE_CLOSE"
                    reason = "REVERSE_POLICY_REVERSE"
            else:
                decision = "NO_ACTION"
                reason = "SIGNAL_FLAT"

        self._record_log(signal, sig_type, pos_side, decision, reason, intent)
        return intent
