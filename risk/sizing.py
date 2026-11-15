from __future__ import annotations
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_FLOOR
import math
from typing import TYPE_CHECKING, Any, Mapping

from quant_backtester.config import InstrumentConfig, RiskConfig, SizingBasis, SizingMode
from quant_backtester.models.order import OrderIntent
from .estimator import RiskEstimate, RiskEstimator

if TYPE_CHECKING:
    from quant_backtester.models.quote import QuoteBar


def _validate_lot_constraints(qty: float, inst: InstrumentConfig) -> float:
    if isinstance(qty, bool) or not isinstance(qty, (int, float)) or not math.isfinite(float(qty)):
        raise ValueError("INVALID_LOT_SIZE")
    q = float(qty)
    if q < inst.min_lot - 1e-9 or q > inst.max_lot + 1e-9:
        raise ValueError("INVALID_LOT_SIZE")
    if inst.lot_step > 0:
        steps = round(q / inst.lot_step)
        if abs(q - steps * inst.lot_step) > 1e-6:
            raise ValueError("INVALID_LOT_STEP")
        return round(steps * inst.lot_step, 8)
    return q


def _floor_to_lot_step(raw_lots: float, lot_step: float) -> float:
    if lot_step <= 0:
        return float(raw_lots)
    d_raw = Decimal(f"{float(raw_lots) + 1e-12:.12f}")
    d_step = Decimal(f"{float(lot_step):.8f}")
    steps = (d_raw / d_step).to_integral_value(rounding=ROUND_FLOOR)
    return float(steps * d_step)


class BasePositionSizer(ABC):
    @abstractmethod
    def size_order(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        instrument_config: InstrumentConfig,
        reference_price: float | None = None,
        quote_bar: QuoteBar | None = None,
        account_currency: str = "USD",
        fx_rates: Mapping[Any, float] | None = None,
    ) -> tuple[float | None, RiskEstimate | None, str | None]:
        raise NotImplementedError


class FixedLotSizer(BasePositionSizer):
    """
    Giữ nguyên quantity đã chỉ định trong OrderIntent (hoặc fallback về instrument.lot_size).
    Kiểm tra tính hợp lệ theo min_lot, max_lot, lot_step.
    """

    def __init__(self, estimator: RiskEstimator | None = None) -> None:
        self.estimator = estimator

    def size_order(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        instrument_config: InstrumentConfig,
        reference_price: float | None = None,
        quote_bar: QuoteBar | None = None,
        account_currency: str = "USD",
        fx_rates: Mapping[Any, float] | None = None,
    ) -> tuple[float | None, RiskEstimate | None, str | None]:
        raw_qty = intent.quantity if intent.quantity is not None else instrument_config.lot_size
        try:
            qty = _validate_lot_constraints(raw_qty, instrument_config)
        except ValueError as exc:
            return None, None, str(exc)
        return qty, None, None


class RiskBudgetPositionSizer(BasePositionSizer):
    """
    Tính khối lượng giao dịch dựa trên ngân sách rủi ro (risk_per_trade_pct hoặc risk_per_trade_amount)
    và khoảng cách Stop Loss + Phí dự kiến từ RiskEstimator.
    Đảm bảo làm tròn xuống theo lot_step và kiểm chứng Step-Down để total_risk <= risk_budget.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        estimator: RiskEstimator,
    ) -> None:
        if (
            risk_config.risk_per_trade_pct is not None
            and risk_config.risk_per_trade_amount is not None
        ):
            raise ValueError("CONFLICTING_RISK_BUDGET_CONFIG")
        self.risk_config = risk_config
        self.estimator = estimator

    def resolve_risk_budget(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
    ) -> float:
        directives = intent.metadata.get("directives", {}) if intent.metadata else {}
        dir_amt = (
            directives.get("risk_per_trade_amount") if isinstance(directives, Mapping) else None
        )
        dir_pct = directives.get("risk_per_trade_pct") if isinstance(directives, Mapping) else None

        if dir_amt is not None and dir_pct is not None:
            raise ValueError("CONFLICTING_RISK_BUDGET_CONFIG")

        if dir_amt is not None:
            amt = float(dir_amt)
            if not math.isfinite(amt) or amt <= 0.0:
                raise ValueError("INVALID_RISK_BUDGET")
            return amt

        if dir_pct is not None:
            pct = float(dir_pct)
            if not math.isfinite(pct) or pct <= 0.0:
                raise ValueError("INVALID_RISK_BUDGET")
            basis_val = (
                current_equity
                if self.risk_config.sizing_basis == SizingBasis.EQUITY
                else current_balance
            )
            return float(basis_val) * pct

        if self.risk_config.risk_per_trade_amount is not None:
            return float(self.risk_config.risk_per_trade_amount)

        if self.risk_config.risk_per_trade_pct is not None:
            basis_val = (
                current_equity
                if self.risk_config.sizing_basis == SizingBasis.EQUITY
                else current_balance
            )
            return float(basis_val) * float(self.risk_config.risk_per_trade_pct)

        raise ValueError("MISSING_RISK_BUDGET")

    def size_order(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        instrument_config: InstrumentConfig,
        reference_price: float | None = None,
        quote_bar: QuoteBar | None = None,
        account_currency: str = "USD",
        fx_rates: Mapping[Any, float] | None = None,
    ) -> tuple[float | None, RiskEstimate | None, str | None]:
        has_sl = intent.stop_loss_price is not None or (
            intent.stop_loss_pct is not None and intent.stop_loss_pct > 0
        )
        if not has_sl:
            return None, None, "STOP_LOSS_REQUIRED_FOR_RISK_SIZING"

        try:
            risk_budget = self.resolve_risk_budget(intent, current_balance, current_equity)
            if not math.isfinite(risk_budget) or risk_budget <= 0.0:
                return None, None, "INVALID_RISK_BUDGET"

            est_one_lot = self.estimator.estimate(
                intent=intent,
                quantity=1.0,
                reference_price=reference_price,
                quote_bar=quote_bar,
                fx_rates=fx_rates,
                instrument_config=instrument_config,
                account_currency=account_currency,
            )
            if est_one_lot.total_risk is None or est_one_lot.stop_distance is None:
                return None, None, "STOP_LOSS_REQUIRED_FOR_RISK_SIZING"
            if est_one_lot.total_risk <= 0.0 or not math.isfinite(est_one_lot.total_risk):
                return None, None, "INVALID_STOP_DISTANCE"

            raw_lots = risk_budget / est_one_lot.total_risk

            if raw_lots > instrument_config.max_lot + 1e-9:
                if self.risk_config.clamp_to_max_lot:
                    raw_lots = instrument_config.max_lot
                else:
                    return None, None, "SIZED_LOT_ABOVE_MAX"

            normalized_lots = _floor_to_lot_step(raw_lots, instrument_config.lot_step)
            if normalized_lots < instrument_config.min_lot - 1e-9:
                return None, None, "SIZED_LOT_BELOW_MIN"

            # Step-Down Verification Loop: đảm bảo estimated_total_risk <= risk_budget ngay cả khi FeeModel có min_fee
            step = instrument_config.lot_step if instrument_config.lot_step > 0 else 0.01
            final_est: RiskEstimate | None = None
            while normalized_lots >= instrument_config.min_lot - 1e-9:
                cand_est = self.estimator.estimate(
                    intent=intent,
                    quantity=normalized_lots,
                    reference_price=reference_price,
                    quote_bar=quote_bar,
                    fx_rates=fx_rates,
                    instrument_config=instrument_config,
                    account_currency=account_currency,
                )
                assert cand_est.total_risk is not None
                if cand_est.total_risk <= risk_budget + 1e-9:
                    final_est = cand_est
                    break
                normalized_lots = _floor_to_lot_step(normalized_lots - step + 1e-12, step)

            if final_est is None or normalized_lots < instrument_config.min_lot - 1e-9:
                return None, None, "SIZED_LOT_BELOW_MIN"

            validated_qty = _validate_lot_constraints(normalized_lots, instrument_config)
            return validated_qty, final_est, None
        except ValueError as exc:
            return None, None, str(exc)


def resolve_position_sizer(
    config: RiskConfig,
    estimator: RiskEstimator | None = None,
) -> BasePositionSizer:
    mode = config.default_sizing_mode
    if mode == SizingMode.RISK_BASED:
        return RiskBudgetPositionSizer(
            risk_config=config,
            estimator=estimator or RiskEstimator(risk_config=config),
        )
    return FixedLotSizer(estimator=estimator)
