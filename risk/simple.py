from __future__ import annotations
from dataclasses import replace
import math
import pandas as pd

from quant_backtester.config import InstrumentConfig, RiskConfig
from quant_backtester.models.order import OrderIntent, OrderSide, OrderType
from .base import BaseRiskManager


class SimpleRiskManager(BaseRiskManager):
    def __init__(
        self,
        config: RiskConfig | None = None,
        instrument_config: InstrumentConfig | None = None,
    ):
        self.config = config if config is not None else RiskConfig()
        self.instrument_config = instrument_config if instrument_config is not None else InstrumentConfig()
        self.last_reject_reason: str | None = None

    def _reject(self, reason: str) -> None:
        self.last_reject_reason = reason
        return None

    def approve_order_intent(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        reference_price: float | None = None,
    ) -> OrderIntent | None:
        self.last_reject_reason = None
        if current_balance <= 0 or current_equity <= 0:
            return self._reject("INSUFFICIENT_EQUITY")

        # D9 / C7: trailing_stop_pct được khai báo trong contract nhưng từ chối với UNSUPPORTED_DIRECTIVE trong V1.0
        if intent.trailing_stop_pct is not None:
            return self._reject("UNSUPPORTED_DIRECTIVE")

        # Rule 5: Không đồng thời khai báo SL theo giá và phần trăm
        if intent.stop_loss_price is not None and intent.stop_loss_pct is not None:
            return self._reject("CONFLICTING_SL_DIRECTIVES")

        # Rule 6: Không đồng thời khai báo TP theo giá và phần trăm
        if intent.take_profit_price is not None and intent.take_profit_pct is not None:
            return self._reject("CONFLICTING_TP_DIRECTIVES")

        # Rule 9 & 11: Kiểm tra hoặc fallback quantity (tính bằng lot)
        inst = self.instrument_config
        qty = intent.quantity if intent.quantity is not None else inst.lot_size
        if not isinstance(qty, (int, float)) or isinstance(qty, bool) or not math.isfinite(float(qty)):
            return self._reject("INVALID_LOT_SIZE")
        qty = float(qty)
        if qty < inst.min_lot - 1e-9 or qty > inst.max_lot + 1e-9:
            return self._reject("INVALID_LOT_SIZE")

        if inst.lot_step > 0:
            steps = round(qty / inst.lot_step)
            if abs(qty - steps * inst.lot_step) > 1e-6:
                return self._reject("INVALID_LOT_STEP")

        # Kiểm tra stop_loss_pct / take_profit_pct hợp lệ nếu có
        sl_pct = intent.stop_loss_pct
        tp_pct = intent.take_profit_pct
        if sl_pct is not None and (not math.isfinite(float(sl_pct)) or float(sl_pct) < 0):
            return self._reject("INVALID_SL_PCT")
        if tp_pct is not None and (not math.isfinite(float(tp_pct)) or float(tp_pct) < 0):
            return self._reject("INVALID_TP_PCT")

        # Rule 7, 8, 10: Kiểm tra chiều SL/TP theo giá tham chiếu (nếu có reference_price hoặc limit_price)
        ref_p = reference_price if reference_price is not None else intent.limit_price
        if intent.stop_loss_price is not None:
            sl_p = float(intent.stop_loss_price)
            if not math.isfinite(sl_p) or sl_p <= 0:
                return self._reject("INVALID_SL_PRICE")
            if ref_p is not None:
                if intent.side == OrderSide.BUY and sl_p >= ref_p:
                    return self._reject("INVALID_SL_SIDE")
                if intent.side == OrderSide.SELL and sl_p <= ref_p:
                    return self._reject("INVALID_SL_SIDE")

        if intent.take_profit_price is not None:
            tp_p = float(intent.take_profit_price)
            if not math.isfinite(tp_p) or tp_p <= 0:
                return self._reject("INVALID_TP_PRICE")
            if ref_p is not None:
                if intent.side == OrderSide.BUY and tp_p <= ref_p:
                    return self._reject("INVALID_TP_SIDE")
                if intent.side == OrderSide.SELL and tp_p >= ref_p:
                    return self._reject("INVALID_TP_SIDE")

        # Rule 11: Fallback về RiskConfig nếu thiếu cả price và pct
        final_sl_pct = sl_pct if (sl_pct is not None or intent.stop_loss_price is not None) else self.config.stop_loss_pct
        final_tp_pct = tp_pct if (tp_pct is not None or intent.take_profit_price is not None) else self.config.take_profit_pct

        return replace(
            intent,
            quantity=qty,
            stop_loss_pct=final_sl_pct,
            take_profit_pct=final_tp_pct,
        )

    def _resolve_sl_tp_prices(
        self,
        side: OrderSide,
        entry_price: float,
        stop_loss_price: float | None,
        take_profit_price: float | None,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
    ) -> tuple[float | None, float | None]:
        cfg = self.config
        eff_sl_pct = stop_loss_pct if (stop_loss_pct is not None or stop_loss_price is not None) else cfg.stop_loss_pct
        eff_tp_pct = take_profit_pct if (take_profit_pct is not None or take_profit_price is not None) else cfg.take_profit_pct

        if side == OrderSide.BUY:
            sl_p = stop_loss_price if stop_loss_price is not None else (entry_price * (1.0 - eff_sl_pct) if eff_sl_pct else None)
            tp_p = take_profit_price if take_profit_price is not None else (entry_price * (1.0 + eff_tp_pct) if eff_tp_pct else None)
        else:
            sl_p = stop_loss_price if stop_loss_price is not None else (entry_price * (1.0 + eff_sl_pct) if eff_sl_pct else None)
            tp_p = take_profit_price if take_profit_price is not None else (entry_price * (1.0 - eff_tp_pct) if eff_tp_pct else None)
        return sl_p, tp_p

    def check_intrabar_exit(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        units: float,
        bar: pd.Series,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> OrderIntent | None:
        sl_price, tp_price = self._resolve_sl_tp_prices(
            side=side,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

        high = float(bar["high"])
        low = float(bar["low"])
        open_price = float(bar["open"])

        # 1. Kiểm tra GAP tại Open
        if side == OrderSide.BUY:
            if sl_price is not None and open_price <= sl_price:
                return OrderIntent(symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET, limit_price=open_price, reason="STOP_GAP")
            if tp_price is not None and open_price >= tp_price:
                return OrderIntent(symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET, limit_price=open_price, reason="TAKE_PROFIT_GAP")

            # 2. Kiểm tra Intrabar High/Low (Quy tắc nếu cả 2 cùng chạm thì ưu tiên SL trước)
            sl_hit = sl_price is not None and low <= sl_price
            tp_hit = tp_price is not None and high >= tp_price

            if sl_hit:
                return OrderIntent(symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET, limit_price=sl_price, reason="STOP_LOSS")
            if tp_hit:
                return OrderIntent(symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET, limit_price=tp_price, reason="TAKE_PROFIT")

        elif side == OrderSide.SELL:
            if sl_price is not None and open_price >= sl_price:
                return OrderIntent(symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, limit_price=open_price, reason="STOP_GAP")
            if tp_price is not None and open_price <= tp_price:
                return OrderIntent(symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, limit_price=open_price, reason="TAKE_PROFIT_GAP")

            sl_hit = sl_price is not None and high >= sl_price
            tp_hit = tp_price is not None and low <= tp_price

            if sl_hit:
                return OrderIntent(symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, limit_price=sl_price, reason="STOP_LOSS")
            if tp_hit:
                return OrderIntent(symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, limit_price=tp_price, reason="TAKE_PROFIT")

        return None
