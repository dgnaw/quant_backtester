from __future__ import annotations
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping

from quant_backtester.config import InstrumentConfig, RiskConfig
from quant_backtester.execution.fee import BaseFeeModel, ZeroFeeModel
from quant_backtester.execution.slippage import BaseSlippageModel
from quant_backtester.models.order import OrderIntent, OrderSide, OrderType

if TYPE_CHECKING:
    from quant_backtester.models.quote import QuoteBar


@dataclass(frozen=True)
class RiskEstimate:
    """
    Kết quả ước tính rủi ro (Mức lỗ dự kiến khi chạm SL + Phí 2 chiều) quy đổi về đồng tiền Account.
    """

    symbol: str
    side: OrderSide
    quantity: float
    units: float
    expected_entry_price: float
    expected_sl_exit_price: float | None
    stop_distance: float | None
    gross_loss_quote_ccy: float | None
    conversion_rate: float
    gross_loss_account_ccy: float | None
    estimated_entry_fee: float
    estimated_exit_fee: float
    total_risk: float | None

    @property
    def total_fee(self) -> float:
        return self.estimated_entry_fee + self.estimated_exit_fee


def resolve_fx_rate(
    from_currency: str,
    to_currency: str,
    fx_rates: Mapping[Any, float] | None = None,
    reference_price: float | None = None,
    symbol: str | None = None,
    base_currency: str | None = None,
) -> float | None:
    """
    Resolve conversion rate from `from_currency` to `to_currency`.
    Supports tuple keys `(src, dst)` and string keys `"GBPUSD"`, `"GBP/USD"`.
    Returns None if conversion is unavailable or invalid.
    """
    src = str(from_currency).strip().upper()
    dst = str(to_currency).strip().upper()
    if src == dst:
        return 1.0

    if fx_rates:
        # Check direct tuple or string keys
        candidates_direct = [
            (src, dst),
            f"{src}{dst}",
            f"{src}/{dst}",
        ]
        for key in candidates_direct:
            if key in fx_rates:
                val = fx_rates[key]
                if (
                    not isinstance(val, bool)
                    and isinstance(val, (int, float))
                    and math.isfinite(float(val))
                    and float(val) > 0.0
                ):
                    return float(val)
                return None

        # Case-insensitive scan
        for k, val in fx_rates.items():
            if isinstance(k, tuple) and len(k) == 2:
                k_src, k_dst = str(k[0]).strip().upper(), str(k[1]).strip().upper()
                if k_src == src and k_dst == dst:
                    if (
                        not isinstance(val, bool)
                        and isinstance(val, (int, float))
                        and math.isfinite(float(val))
                        and float(val) > 0.0
                    ):
                        return float(val)
                    return None
                if k_src == dst and k_dst == src:
                    if (
                        not isinstance(val, bool)
                        and isinstance(val, (int, float))
                        and math.isfinite(float(val))
                        and float(val) > 0.0
                    ):
                        return 1.0 / float(val)
                    return None
            elif isinstance(k, str):
                k_norm = k.strip().upper().replace("/", "")
                if k_norm == f"{src}{dst}":
                    if (
                        not isinstance(val, bool)
                        and isinstance(val, (int, float))
                        and math.isfinite(float(val))
                        and float(val) > 0.0
                    ):
                        return float(val)
                    return None
                if k_norm == f"{dst}{src}":
                    if (
                        not isinstance(val, bool)
                        and isinstance(val, (int, float))
                        and math.isfinite(float(val))
                        and float(val) > 0.0
                    ):
                        return 1.0 / float(val)
                    return None

    # Fallback: if base_currency == to_currency and symbol == dst+src (e.g. USDJPY on USD account),
    # reference_price is dst/src so src->dst is 1 / reference_price
    if (
        base_currency is not None
        and str(base_currency).strip().upper() == dst
        and reference_price is not None
        and reference_price > 0.0
        and symbol is not None
        and symbol.strip().upper().replace("/", "") == f"{dst}{src}"
    ):
        return 1.0 / float(reference_price)

    return None


class RiskEstimator:
    """
    Ước tính mức rủi ro tiềm năng của một OrderIntent theo khoảng cách Stop Loss,
    chi phí giao dịch dự kiến (Entry + Exit Fee) và quy đổi tiền tệ về Account Currency.
    """

    def __init__(
        self,
        instrument_config: InstrumentConfig | None = None,
        account_currency: str = "USD",
        fee_model: BaseFeeModel | None = None,
        slippage_model: BaseSlippageModel | None = None,
        include_slippage_in_risk: bool = False,
        default_stop_loss_pct: float = 0.0,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.risk_config = risk_config
        self.instrument_config = (
            instrument_config if instrument_config is not None else InstrumentConfig()
        )
        self.account_currency = str(account_currency).strip().upper()
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        self.include_slippage_in_risk = (
            bool(risk_config.include_slippage_in_risk)
            if risk_config is not None
            else bool(include_slippage_in_risk)
        )
        self.include_fees_in_risk = (
            bool(risk_config.include_fees_in_risk) if risk_config is not None else True
        )
        self.default_stop_loss_pct = (
            float(risk_config.stop_loss_pct)
            if risk_config is not None
            else float(default_stop_loss_pct)
        )

    def resolve_expected_entry_price(
        self,
        intent: OrderIntent,
        reference_price: float | None = None,
        quote_bar: QuoteBar | None = None,
    ) -> float | None:
        base_p: float | None = None
        if intent.order_type == OrderType.LIMIT and intent.limit_price is not None:
            base_p = intent.limit_price
        elif intent.order_type == OrderType.STOP_MARKET and intent.stop_price is not None:
            base_p = intent.stop_price
        elif reference_price is not None:
            base_p = reference_price
        elif quote_bar is not None:
            base_p = quote_bar.ask_close if intent.side == OrderSide.BUY else quote_bar.bid_close
        elif intent.limit_price is not None:
            base_p = intent.limit_price
        elif intent.stop_price is not None:
            base_p = intent.stop_price

        if base_p is None:
            return None
        if (
            isinstance(base_p, bool)
            or not isinstance(base_p, (int, float))
            or not math.isfinite(float(base_p))
            or float(base_p) <= 0.0
        ):
            return None

        entry_p = float(base_p)
        if (
            self.include_slippage_in_risk
            and self.slippage_model is not None
            and intent.order_type != OrderType.LIMIT
        ):
            entry_p = float(self.slippage_model.calculate_fill_price(entry_p, intent.side))
        return entry_p

    def resolve_entry_price(
        self,
        intent: OrderIntent,
        reference_price: float | None = None,
        quote_bar: QuoteBar | None = None,
    ) -> float:
        entry_p = self.resolve_expected_entry_price(
            intent, reference_price=reference_price, quote_bar=quote_bar
        )
        if entry_p is None:
            raise ValueError("MISSING_REFERENCE_PRICE")
        return entry_p

    def resolve_sl_exit_price(
        self, intent: OrderIntent, expected_entry_price: float
    ) -> float | None:
        if intent.stop_loss_price is not None and intent.stop_loss_pct is not None:
            raise ValueError("CONFLICTING_SL_DIRECTIVES")

        sl_p: float | None = None
        if intent.stop_loss_price is not None:
            raw_sl = intent.stop_loss_price
            if (
                isinstance(raw_sl, bool)
                or not isinstance(raw_sl, (int, float))
                or not math.isfinite(float(raw_sl))
                or float(raw_sl) <= 0.0
            ):
                raise ValueError("INVALID_SL_PRICE")
            sl_p = float(raw_sl)
        else:
            eff_pct = (
                intent.stop_loss_pct
                if intent.stop_loss_pct is not None
                else (self.default_stop_loss_pct if self.default_stop_loss_pct > 0.0 else None)
            )
            if eff_pct is not None:
                if (
                    isinstance(eff_pct, bool)
                    or not isinstance(eff_pct, (int, float))
                    or not math.isfinite(float(eff_pct))
                    or float(eff_pct) < 0.0
                ):
                    raise ValueError("INVALID_SL_PCT")
                if float(eff_pct) > 0.0:
                    pct = float(eff_pct)
                    sl_p = (
                        expected_entry_price * (1.0 - pct)
                        if intent.side == OrderSide.BUY
                        else expected_entry_price * (1.0 + pct)
                    )

        if sl_p is None:
            return None

        if sl_p <= 0.0 or not math.isfinite(sl_p):
            raise ValueError("INVALID_SL_PRICE")

        if intent.side == OrderSide.BUY and sl_p >= expected_entry_price - 1e-12:
            raise ValueError("INVALID_SL_SIDE")
        if intent.side == OrderSide.SELL and sl_p <= expected_entry_price + 1e-12:
            raise ValueError("INVALID_SL_SIDE")

        sl_exit = sl_p
        if self.include_slippage_in_risk and self.slippage_model is not None:
            close_side = OrderSide.SELL if intent.side == OrderSide.BUY else OrderSide.BUY
            sl_exit = float(self.slippage_model.calculate_fill_price(sl_p, close_side))

        return sl_exit

    def estimate(
        self,
        intent: OrderIntent,
        quantity: float,
        reference_price: float | None = None,
        quote_bar: QuoteBar | None = None,
        fx_rates: Mapping[Any, float] | None = None,
        instrument_config: InstrumentConfig | None = None,
        account_currency: str | None = None,
    ) -> RiskEstimate:
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, (int, float))
            or not math.isfinite(float(quantity))
            or float(quantity) <= 0.0
        ):
            raise ValueError("INVALID_LOT_SIZE")

        inst = instrument_config if instrument_config is not None else self.instrument_config
        acc_ccy = (
            str(account_currency).strip().upper()
            if account_currency is not None
            else self.account_currency
        )
        qty = float(quantity)
        units = inst.lots_to_units(qty)

        expected_entry = self.resolve_entry_price(
            intent, reference_price=reference_price, quote_bar=quote_bar
        )
        expected_sl_exit = self.resolve_sl_exit_price(intent, expected_entry)

        conv_rate = resolve_fx_rate(
            inst.quote_currency,
            acc_ccy,
            fx_rates,
            reference_price=expected_entry,
            symbol=inst.symbol,
            base_currency=inst.base_currency,
        )
        if conv_rate is None or conv_rate <= 0.0:
            raise ValueError("CURRENCY_CONVERSION_UNAVAILABLE")

        entry_fee = 0.0
        exit_fee = 0.0
        if self.include_fees_in_risk and self.fee_model is not None:
            fee_ccy = getattr(self.fee_model, "currency", acc_ccy)
            fee_conv_rate = resolve_fx_rate(
                fee_ccy,
                acc_ccy,
                fx_rates,
                reference_price=expected_entry,
                symbol=inst.symbol,
                base_currency=inst.base_currency,
            )
            if fee_conv_rate is None or fee_conv_rate <= 0.0:
                raise ValueError("CURRENCY_CONVERSION_UNAVAILABLE")

            raw_entry_fee = self.fee_model.calculate_fee(
                expected_entry,
                units,
                quantity=qty,
                side=intent.side,
                is_opening=True,
            )
            close_side = OrderSide.SELL if intent.side == OrderSide.BUY else OrderSide.BUY
            exit_ref_p = expected_sl_exit if expected_sl_exit is not None else expected_entry
            raw_exit_fee = self.fee_model.calculate_fee(
                exit_ref_p,
                units,
                quantity=qty,
                side=close_side,
                is_opening=False,
            )
            entry_fee = float(raw_entry_fee) * fee_conv_rate
            exit_fee = float(raw_exit_fee) * fee_conv_rate

        if expected_sl_exit is None:
            return RiskEstimate(
                symbol=intent.symbol,
                side=intent.side,
                quantity=qty,
                units=units,
                expected_entry_price=expected_entry,
                expected_sl_exit_price=None,
                stop_distance=None,
                gross_loss_quote_ccy=None,
                conversion_rate=conv_rate,
                gross_loss_account_ccy=None,
                estimated_entry_fee=entry_fee,
                estimated_exit_fee=exit_fee,
                total_risk=None,
            )

        stop_dist = (
            (expected_entry - expected_sl_exit)
            if intent.side == OrderSide.BUY
            else (expected_sl_exit - expected_entry)
        )
        if stop_dist <= 0.0 or not math.isfinite(stop_dist):
            raise ValueError("INVALID_SL_SIDE")

        gross_loss_quote = stop_dist * units
        gross_loss_acc = gross_loss_quote * conv_rate
        total_risk = gross_loss_acc + entry_fee + exit_fee

        return RiskEstimate(
            symbol=intent.symbol,
            side=intent.side,
            quantity=qty,
            units=units,
            expected_entry_price=expected_entry,
            expected_sl_exit_price=expected_sl_exit,
            stop_distance=stop_dist,
            gross_loss_quote_ccy=gross_loss_quote,
            conversion_rate=conv_rate,
            gross_loss_account_ccy=gross_loss_acc,
            estimated_entry_fee=entry_fee,
            estimated_exit_fee=exit_fee,
            total_risk=total_risk,
        )

    def estimate_trade_risk(
        self,
        intent: OrderIntent,
        quantity: float,
        instrument_config: InstrumentConfig | None = None,
        reference_price: float | None = None,
        quote_bar: QuoteBar | None = None,
        account_currency: str = "USD",
        fx_rates: Mapping[Any, float] | None = None,
    ) -> tuple[RiskEstimate | None, str | None]:
        """Tuple-returning wrapper around `estimate` used by Sizers and SimpleRiskManager."""
        try:
            est = self.estimate(
                intent=intent,
                quantity=quantity,
                reference_price=reference_price,
                quote_bar=quote_bar,
                fx_rates=fx_rates,
                instrument_config=instrument_config,
                account_currency=account_currency,
            )
            if est.total_risk is None:
                return None, "STOP_LOSS_REQUIRED"
            return est, None
        except ValueError as exc:
            return None, str(exc)
