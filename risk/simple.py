"""
SimpleRiskManager: Full Group D & Group F Risk Gatekeeper implementation.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_backtester.config import InstrumentConfig, RiskConfig, SizingBasis
from quant_backtester.data.trading_calendar import TradingCalendar
from quant_backtester.models.order import OrderIntent, OrderSide, OrderType
from quant_backtester.portfolio.position import PositionSide
from quant_backtester.risk.audit import RiskAuditLog
from quant_backtester.risk.base import BaseRiskManager, RiskContext
from quant_backtester.risk.decision import RiskCheckResult, RiskDecision, RiskDecisionStatus
from quant_backtester.risk.estimator import RiskEstimate, RiskEstimator
from quant_backtester.risk.halt import HaltReason, TradingHaltManager
from quant_backtester.risk.margin import LeverageMarginModel, MarginSnapshot
from quant_backtester.risk.monitors import (
    CorrelationRiskMonitor,
    DailyLossMonitor,
    DrawdownMonitor,
    ExposureMonitor,
)
from quant_backtester.risk.sizing import resolve_position_sizer

if TYPE_CHECKING:
    from quant_backtester.execution.fee import BaseFeeModel
    from quant_backtester.execution.slippage import BaseSlippageModel
    from quant_backtester.portfolio.position import Position, PositionView


EXIT_INTENT_REASONS = frozenset(
    {
        "STRATEGY_EXIT",
        "PROTECTIVE_SL",
        "PROTECTIVE_TP",
        "STOP_LOSS",
        "TAKE_PROFIT",
        "STOP_GAP",
        "TAKE_PROFIT_GAP",
        "END_OF_TEST",
        "MARGIN_STOP_OUT",
        "MAX_DRAWDOWN_LIQUIDATION",
    }
)


class SimpleRiskManager(BaseRiskManager):
    """
    Production Risk Gatekeeper implementing Groups C, D, and F:
    - Position Sizing (Fixed Lot & Risk-Based with Step-Down Fee/Slippage Loop)
    - Per-Trade Risk Cap
    - Daily Loss Limit (with TradingCalendar session reset & deposit adjustment)
    - Maximum Drawdown Circuit Breaker (HWM with deposit adjustment)
    - Symbol & Account Exposure Limits (including Pending Entry Order reservation)
    - Leverage Margin Model, Margin Call & Stop-Out selection
    - Pairwise Directional Correlation Risk Control
    - Trading Halt Manager & Exit Order Bypass Validation
    - Structured RiskDecision & RiskAuditLog
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        instrument_config: InstrumentConfig | None = None,
        account_currency: str = "USD",
        default_leverage: float = 100.0,
        fee_model: BaseFeeModel | None = None,
        slippage_model: BaseSlippageModel | None = None,
        calendar: TradingCalendar | None = None,
        instrument_by_symbol: dict[str, InstrumentConfig] | None = None,
    ) -> None:
        self.config = config if config is not None else RiskConfig()
        self.instrument_config = (
            instrument_config if instrument_config is not None else InstrumentConfig()
        )
        self.account_currency = account_currency
        self.default_leverage = (
            self.config.leverage if self.config.leverage is not None else default_leverage
        )
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        self.instrument_by_symbol: dict[str, InstrumentConfig] = (
            dict(instrument_by_symbol) if instrument_by_symbol is not None else {}
        )
        if self.instrument_config.symbol not in self.instrument_by_symbol:
            self.instrument_by_symbol[self.instrument_config.symbol] = self.instrument_config

        self.audit_log = RiskAuditLog()
        self.halt_manager = TradingHaltManager(audit_log=self.audit_log)
        self.estimator = RiskEstimator(
            fee_model=self.fee_model,
            slippage_model=self.slippage_model,
            risk_config=self.config,
        )
        self.sizer = resolve_position_sizer(self.config, self.estimator)
        self.margin_model = LeverageMarginModel(
            risk_config=self.config,
            default_leverage=self.default_leverage,
        )
        self.daily_loss_monitor = DailyLossMonitor(
            risk_config=self.config,
            calendar=calendar,
        )
        self.drawdown_monitor = DrawdownMonitor(risk_config=self.config)
        self.exposure_monitor = ExposureMonitor(risk_config=self.config)
        self.correlation_monitor = CorrelationRiskMonitor(risk_config=self.config)

        self.last_reject_reason: str | None = None
        self.last_decision: RiskDecision | None = None
        self.last_margin_snapshot: MarginSnapshot | None = None
        self._calendar = calendar

    def reset_for_run(self) -> None:
        """
        Reset per-run state (DailyLossMonitor, DrawdownMonitor, transient halts, and AuditLog)
        while preserving any pre-configured HALTED_MANUAL or HALTED_SYSTEM_ERROR state.
        """
        preserved_reasons = [
            r
            for r in self.halt_manager.active_reasons
            if r in (HaltReason.HALTED_MANUAL, HaltReason.HALTED_SYSTEM_ERROR)
        ]
        self.audit_log.clear()
        self.halt_manager.reset()
        for r in preserved_reasons:
            self.halt_manager.trigger_halt(r, details="Pre-configured halt before run")
        self.daily_loss_monitor = DailyLossMonitor(
            risk_config=self.config,
            calendar=self._calendar,
        )
        self.drawdown_monitor = DrawdownMonitor(risk_config=self.config)
        self.last_reject_reason = None
        self.last_decision = None
        self.last_margin_snapshot = None

    def _resolve_instrument(
        self,
        symbol: str,
        context: RiskContext | None = None,
    ) -> InstrumentConfig:
        if context is not None and symbol in context.instrument_by_symbol:
            return context.instrument_by_symbol[symbol]
        if symbol in self.instrument_by_symbol:
            return self.instrument_by_symbol[symbol]
        return self.instrument_config

    def _record_decision(
        self,
        decision: RiskDecision,
        intent: OrderIntent,
        timestamp: pd.Timestamp | None = None,
    ) -> RiskDecision:
        self.last_decision = decision
        self.last_reject_reason = decision.reject_reason
        self.audit_log.record_decision(
            timestamp=timestamp,
            symbol=intent.symbol,
            side=intent.side.value if hasattr(intent.side, "value") else str(intent.side),
            decision=decision,
        )
        return decision

    def _build_rejection(
        self,
        intent: OrderIntent,
        reject_code: str,
        check_name: str = "RISK_VALIDATION",
        message: str = "",
        checks: list[RiskCheckResult] | None = None,
        timestamp: pd.Timestamp | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
        estimated_risk_amount: float | None = None,
        required_margin: float | None = None,
    ) -> RiskDecision:
        all_checks = list(checks) if checks is not None else []
        if not any(not c.passed and c.reject_code == reject_code for c in all_checks):
            all_checks.append(
                RiskCheckResult(
                    check_name=check_name,
                    passed=False,
                    reject_code=reject_code,
                    message=message or reject_code,
                    current_value=current_value,
                    limit_value=limit_value,
                )
            )
        decision = RiskDecision(
            status=RiskDecisionStatus.REJECTED,
            approved_intent=None,
            reject_reason=reject_code,
            checks=tuple(all_checks),
            sized_quantity=None,
            estimated_risk_amount=estimated_risk_amount,
            required_margin=required_margin,
        )
        return self._record_decision(decision, intent, timestamp=timestamp)

    def update_portfolio_state(self, context: RiskContext) -> MarginSnapshot:
        """
        Update intraday portfolio risk monitors (Daily Loss, Max Drawdown, Margin Call)
        and return the latest MarginSnapshot.
        """
        inst_map = {**self.instrument_by_symbol, **(context.instrument_by_symbol or {})}
        account_ccy = context.account_currency or self.account_currency

        self.daily_loss_monitor.update(
            timestamp=context.timestamp,
            current_equity=context.current_equity,
            net_deposits=context.net_deposits,
            halt_manager=self.halt_manager,
        )
        self.drawdown_monitor.update(
            timestamp=context.timestamp,
            current_equity=context.current_equity,
            net_deposits=context.net_deposits,
            halt_manager=self.halt_manager,
        )

        snapshot = self.margin_model.compute_snapshot(
            equity=context.current_equity,
            active_positions=context.active_positions,
            pending_entry_orders=context.pending_entry_orders,
            mark_prices=context.mark_prices,
            instrument_by_symbol=inst_map,
            default_instrument=self.instrument_config,
            account_currency=account_ccy,
            fx_rates=context.fx_rates,
        )
        if snapshot.is_margin_call:
            self.halt_manager.trigger_halt(
                HaltReason.HALTED_MARGIN_CALL,
                timestamp=context.timestamp,
                details=f"margin_level_pct={snapshot.margin_level_pct:.2f}%",
            )
        elif self.halt_manager.has_reason(HaltReason.HALTED_MARGIN_CALL):
            self.halt_manager.clear_halt(HaltReason.HALTED_MARGIN_CALL)

        self.last_margin_snapshot = snapshot
        return snapshot

    def validate_exit_intent(
        self,
        intent: OrderIntent,
        active_position: Position | PositionView | None,
        instrument_config: InstrumentConfig | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> RiskDecision:
        """
        Validate a risk-reducing exit intent (Task F8 - DF9).

        Exit orders bypass Trading Halt, Daily Loss, Max Drawdown, Margin Call,
        Exposure, and Correlation checks, but MUST strictly reduce or close an
        existing open position without reversing it.
        """
        if active_position is None or active_position.is_flat or active_position.quantity <= 0.0:
            return self._build_rejection(
                intent=intent,
                reject_code="NO_OPEN_POSITION_TO_EXIT",
                check_name="EXIT_VALIDATION",
                message=f"Cannot exit {intent.symbol}: no open position.",
                timestamp=timestamp,
            )

        is_opposite = (
            (active_position.side == PositionSide.LONG and intent.side == OrderSide.SELL)
            or (active_position.side == PositionSide.SHORT and intent.side == OrderSide.BUY)
        )
        if not is_opposite:
            return self._build_rejection(
                intent=intent,
                reject_code="EXIT_ORDER_SAME_SIDE_AS_POSITION",
                check_name="EXIT_VALIDATION",
                message=(
                    f"Exit order side {intent.side} matches open position side "
                    f"{active_position.side}; would increase exposure."
                ),
                timestamp=timestamp,
            )

        qty = intent.quantity if intent.quantity is not None else active_position.quantity
        if (
            not isinstance(qty, (int, float))
            or isinstance(qty, bool)
            or not math.isfinite(float(qty))
            or float(qty) <= 0.0
        ):
            return self._build_rejection(
                intent=intent,
                reject_code="INVALID_LOT_SIZE",
                check_name="EXIT_VALIDATION",
                message=f"Invalid exit quantity {qty}.",
                timestamp=timestamp,
            )
        qty = float(qty)

        if qty > active_position.quantity + 1e-9:
            return self._build_rejection(
                intent=intent,
                reject_code="EXIT_QUANTITY_EXCEEDS_POSITION",
                check_name="EXIT_VALIDATION",
                message=(
                    f"Exit quantity {qty:.4f} exceeds open position quantity "
                    f"{active_position.quantity:.4f}; reversal not permitted via exit bypass."
                ),
                timestamp=timestamp,
                current_value=qty,
                limit_value=active_position.quantity,
            )

        approved = replace(intent, quantity=qty)
        check = RiskCheckResult(
            check_name="EXIT_VALIDATION",
            passed=True,
            message="Exit order validated (risk-reducing).",
            current_value=qty,
            limit_value=active_position.quantity,
        )
        decision = RiskDecision(
            status=RiskDecisionStatus.APPROVED,
            approved_intent=approved,
            reject_reason=None,
            checks=(check,),
            sized_quantity=qty,
            estimated_risk_amount=0.0,
            required_margin=0.0,
        )
        return self._record_decision(decision, intent, timestamp=timestamp)

    def _compute_max_allowed_trade_risk(
        self,
        current_balance: float,
        current_equity: float,
    ) -> float | None:
        caps: list[float] = []
        if self.config.max_risk_per_trade_pct is not None:
            basis = (
                float(current_equity)
                if self.config.sizing_basis == SizingBasis.EQUITY
                else float(current_balance)
            )
            caps.append(max(0.0, basis * float(self.config.max_risk_per_trade_pct)))
        if self.config.max_risk_per_trade_amount is not None:
            caps.append(float(self.config.max_risk_per_trade_amount))
        if not caps:
            return None
        return min(caps)

    def evaluate_order_intent(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        reference_price: float | None = None,
        context: RiskContext | None = None,
    ) -> RiskDecision:
        """
        Full Group D + Group F evaluation returning a structured RiskDecision.
        """
        self.last_reject_reason = None
        ts = context.timestamp if context is not None else None
        inst = self._resolve_instrument(intent.symbol, context)
        inst_map = {
            **self.instrument_by_symbol,
            **(context.instrument_by_symbol if context is not None else {}),
            intent.symbol: inst,
        }
        account_ccy = (
            context.account_currency
            if (context is not None and context.account_currency)
            else self.account_currency
        )
        fx_rates = context.fx_rates if context is not None else None
        quote_bar = context.quote_bar if context is not None else None
        active_positions = context.active_positions if context is not None else {}
        pending_orders = context.pending_entry_orders if context is not None else []
        mark_prices = context.mark_prices if context is not None else {}

        # Route explicit exit intents through validate_exit_intent if position state is provided
        if intent.reason in EXIT_INTENT_REASONS and context is not None:
            pos = active_positions.get(intent.symbol)
            return self.validate_exit_intent(
                intent=intent,
                active_position=pos,
                instrument_config=inst,
                timestamp=ts,
            )

        checks: list[RiskCheckResult] = []

        # 1. Update monitors if enabled
        if self.daily_loss_monitor.is_enabled:
            dl_res = self.daily_loss_monitor.update(
                timestamp=ts,
                current_equity=current_equity,
                net_deposits=context.net_deposits if context is not None else 0.0,
                halt_manager=self.halt_manager,
            )
            checks.append(dl_res)

        if self.drawdown_monitor.is_enabled:
            dd_res = self.drawdown_monitor.update(
                timestamp=ts,
                current_equity=current_equity,
                net_deposits=context.net_deposits if context is not None else 0.0,
                halt_manager=self.halt_manager,
            )
            checks.append(dd_res)

        # Update margin call state if margin call threshold is configured
        if self.config.margin_call_level_pct is not None and current_equity > 0:
            try:
                mc_snap = self.margin_model.compute_snapshot(
                    equity=current_equity,
                    active_positions=active_positions,
                    pending_entry_orders=pending_orders,
                    mark_prices=mark_prices,
                    instrument_by_symbol=inst_map,
                    default_instrument=inst,
                    account_currency=account_ccy,
                    fx_rates=fx_rates,
                )
                self.last_margin_snapshot = mc_snap
                if mc_snap.is_margin_call:
                    self.halt_manager.trigger_halt(
                        HaltReason.HALTED_MARGIN_CALL,
                        timestamp=ts,
                        details=f"margin_level_pct={mc_snap.margin_level_pct:.2f}%",
                    )
                elif self.halt_manager.has_reason(HaltReason.HALTED_MARGIN_CALL):
                    self.halt_manager.clear_halt(HaltReason.HALTED_MARGIN_CALL)
            except ValueError as exc:
                return self._build_rejection(
                    intent=intent,
                    reject_code=str(exc),
                    check_name="MARGIN_CHECK",
                    checks=checks,
                    timestamp=ts,
                )

        # 2. Check Trading Halt state
        if self.halt_manager.is_halted:
            reject_code = self.halt_manager.primary_reject_code or "TRADING_HALTED"
            return self._build_rejection(
                intent=intent,
                reject_code=reject_code,
                check_name="TRADING_HALT",
                message=f"Trading halted: {[r.value for r in self.halt_manager.active_reasons]}",
                checks=checks,
                timestamp=ts,
            )

        # 3. Basic Equity & Balance check
        if current_balance <= 0 or current_equity <= 0:
            return self._build_rejection(
                intent=intent,
                reject_code="INSUFFICIENT_EQUITY",
                check_name="EQUITY_CHECK",
                checks=checks,
                timestamp=ts,
            )

        # 4. Directive & SL/TP sanity checks (Group D contract)
        if intent.trailing_stop_pct is not None:
            return self._build_rejection(
                intent=intent,
                reject_code="UNSUPPORTED_DIRECTIVE",
                check_name="DIRECTIVE_CHECK",
                checks=checks,
                timestamp=ts,
            )

        if intent.stop_loss_price is not None and intent.stop_loss_pct is not None:
            return self._build_rejection(
                intent=intent,
                reject_code="CONFLICTING_SL_DIRECTIVES",
                check_name="SL_DIRECTIVE_CHECK",
                checks=checks,
                timestamp=ts,
            )

        if intent.take_profit_price is not None and intent.take_profit_pct is not None:
            return self._build_rejection(
                intent=intent,
                reject_code="CONFLICTING_TP_DIRECTIVES",
                check_name="TP_DIRECTIVE_CHECK",
                checks=checks,
                timestamp=ts,
            )

        sl_pct = intent.stop_loss_pct
        tp_pct = intent.take_profit_pct
        if sl_pct is not None and (not math.isfinite(float(sl_pct)) or float(sl_pct) < 0):
            return self._build_rejection(
                intent=intent,
                reject_code="INVALID_SL_PCT",
                check_name="SL_DIRECTIVE_CHECK",
                checks=checks,
                timestamp=ts,
            )
        if tp_pct is not None and (not math.isfinite(float(tp_pct)) or float(tp_pct) < 0):
            return self._build_rejection(
                intent=intent,
                reject_code="INVALID_TP_PCT",
                check_name="TP_DIRECTIVE_CHECK",
                checks=checks,
                timestamp=ts,
            )

        ref_p = (
            reference_price
            if reference_price is not None
            else (
                context.reference_price
                if (context is not None and context.reference_price is not None)
                else intent.limit_price
            )
        )

        if intent.stop_loss_price is not None:
            sl_p = float(intent.stop_loss_price)
            if not math.isfinite(sl_p) or sl_p <= 0:
                return self._build_rejection(
                    intent=intent,
                    reject_code="INVALID_SL_PRICE",
                    check_name="SL_DIRECTIVE_CHECK",
                    checks=checks,
                    timestamp=ts,
                )
            if ref_p is not None:
                if intent.side == OrderSide.BUY and sl_p >= ref_p:
                    return self._build_rejection(
                        intent=intent,
                        reject_code="INVALID_SL_SIDE",
                        check_name="SL_DIRECTIVE_CHECK",
                        checks=checks,
                        timestamp=ts,
                    )
                if intent.side == OrderSide.SELL and sl_p <= ref_p:
                    return self._build_rejection(
                        intent=intent,
                        reject_code="INVALID_SL_SIDE",
                        check_name="SL_DIRECTIVE_CHECK",
                        checks=checks,
                        timestamp=ts,
                    )

        if intent.take_profit_price is not None:
            tp_p = float(intent.take_profit_price)
            if not math.isfinite(tp_p) or tp_p <= 0:
                return self._build_rejection(
                    intent=intent,
                    reject_code="INVALID_TP_PRICE",
                    check_name="TP_DIRECTIVE_CHECK",
                    checks=checks,
                    timestamp=ts,
                )
            if ref_p is not None:
                if intent.side == OrderSide.BUY and tp_p <= ref_p:
                    return self._build_rejection(
                        intent=intent,
                        reject_code="INVALID_TP_SIDE",
                        check_name="TP_DIRECTIVE_CHECK",
                        checks=checks,
                        timestamp=ts,
                    )
                if intent.side == OrderSide.SELL and tp_p >= ref_p:
                    return self._build_rejection(
                        intent=intent,
                        reject_code="INVALID_TP_SIDE",
                        check_name="TP_DIRECTIVE_CHECK",
                        checks=checks,
                        timestamp=ts,
                    )

        final_sl_pct = (
            sl_pct
            if (sl_pct is not None or intent.stop_loss_price is not None)
            else self.config.stop_loss_pct
        )
        final_tp_pct = (
            tp_pct
            if (tp_pct is not None or intent.take_profit_price is not None)
            else self.config.take_profit_pct
        )

        working_intent = replace(
            intent,
            stop_loss_pct=final_sl_pct,
            take_profit_pct=final_tp_pct,
        )

        # 5. Position Sizing (Task F1)
        qty, risk_estimate, size_reject = self.sizer.size_order(
            intent=working_intent,
            current_balance=current_balance,
            current_equity=current_equity,
            instrument_config=inst,
            reference_price=ref_p,
            quote_bar=quote_bar,
            account_currency=account_ccy,
            fx_rates=fx_rates,
        )
        if size_reject is not None or qty is None:
            return self._build_rejection(
                intent=intent,
                reject_code=size_reject or "INVALID_LOT_SIZE",
                check_name="POSITION_SIZING",
                checks=checks,
                timestamp=ts,
                estimated_risk_amount=risk_estimate.total_risk if risk_estimate else None,
            )

        working_intent = replace(working_intent, quantity=qty)
        checks.append(
            RiskCheckResult(
                check_name="POSITION_SIZING",
                passed=True,
                current_value=qty,
            )
        )

        # 6. Per-Trade Risk Cap (Task F2)
        max_allowed_risk = self._compute_max_allowed_trade_risk(
            current_balance=current_balance,
            current_equity=current_equity,
        )
        if max_allowed_risk is not None:
            has_sl = working_intent.stop_loss_price is not None or (
                working_intent.stop_loss_pct is not None and working_intent.stop_loss_pct > 0
            )
            if not has_sl:
                return self._build_rejection(
                    intent=working_intent,
                    reject_code="STOP_LOSS_REQUIRED_BY_RISK_CAP",
                    check_name="PER_TRADE_RISK_CAP",
                    message="Per-trade risk cap requires a valid Stop Loss.",
                    checks=checks,
                    timestamp=ts,
                )

            if risk_estimate is None:
                risk_estimate, est_err = self.estimator.estimate_trade_risk(
                    intent=working_intent,
                    quantity=qty,
                    instrument_config=inst,
                    reference_price=ref_p,
                    quote_bar=quote_bar,
                    account_currency=account_ccy,
                    fx_rates=fx_rates,
                )
                if est_err is not None or risk_estimate is None:
                    code = (
                        "STOP_LOSS_REQUIRED_BY_RISK_CAP"
                        if est_err == "STOP_LOSS_REQUIRED"
                        else (est_err or "INVALID_SL_PRICE")
                    )
                    return self._build_rejection(
                        intent=working_intent,
                        reject_code=code,
                        check_name="PER_TRADE_RISK_CAP",
                        checks=checks,
                        timestamp=ts,
                    )

            if risk_estimate.total_risk > max_allowed_risk + 1e-9:
                return self._build_rejection(
                    intent=working_intent,
                    reject_code="MAX_TRADE_RISK_EXCEEDED",
                    check_name="PER_TRADE_RISK_CAP",
                    message=(
                        f"Estimated trade risk {risk_estimate.total_risk:.2f} exceeds "
                        f"max allowed risk {max_allowed_risk:.2f}."
                    ),
                    checks=checks,
                    timestamp=ts,
                    current_value=risk_estimate.total_risk,
                    limit_value=max_allowed_risk,
                    estimated_risk_amount=risk_estimate.total_risk,
                )

            checks.append(
                RiskCheckResult(
                    check_name="PER_TRADE_RISK_CAP",
                    passed=True,
                    current_value=risk_estimate.total_risk,
                    limit_value=max_allowed_risk,
                )
            )

        # Resolve expected entry price for exposure, correlation, and margin checks
        if risk_estimate is not None:
            expected_entry_price = risk_estimate.expected_entry_price
        else:
            resolved_p = self.estimator.resolve_expected_entry_price(
                intent=working_intent,
                reference_price=ref_p,
                quote_bar=quote_bar,
            )
            if resolved_p is None and working_intent.symbol in mark_prices:
                resolved_p = mark_prices[working_intent.symbol]
            expected_entry_price = float(resolved_p) if resolved_p is not None else 0.0

        # 7. Exposure Limits (Task F5)
        if self.exposure_monitor.is_enabled:
            exp_res = self.exposure_monitor.check_exposure(
                intent=working_intent,
                quantity=qty,
                expected_entry_price=expected_entry_price,
                active_positions=active_positions,
                pending_entry_orders=pending_orders,
                mark_prices=mark_prices,
                instrument_by_symbol=inst_map,
                default_instrument=inst,
                account_currency=account_ccy,
                fx_rates=fx_rates,
            )
            checks.append(exp_res)
            if not exp_res.passed:
                return self._build_rejection(
                    intent=working_intent,
                    reject_code=exp_res.reject_code or "EXPOSURE_LIMIT_EXCEEDED",
                    check_name="EXPOSURE_LIMIT",
                    message=exp_res.message,
                    checks=checks,
                    timestamp=ts,
                    current_value=exp_res.current_value,
                    limit_value=exp_res.limit_value,
                    estimated_risk_amount=risk_estimate.total_risk if risk_estimate else None,
                )

        # 8. Correlation Risk (Task F7)
        if self.correlation_monitor.is_enabled:
            corr_res = self.correlation_monitor.check_correlation(
                intent=working_intent,
                quantity=qty,
                expected_entry_price=expected_entry_price,
                active_positions=active_positions,
                pending_entry_orders=pending_orders,
                mark_prices=mark_prices,
                instrument_by_symbol=inst_map,
                default_instrument=inst,
                account_currency=account_ccy,
                fx_rates=fx_rates,
            )
            checks.append(corr_res)
            if not corr_res.passed:
                return self._build_rejection(
                    intent=working_intent,
                    reject_code=corr_res.reject_code or "CORRELATION_RISK_EXCEEDED",
                    check_name="CORRELATION_RISK",
                    message=corr_res.message,
                    checks=checks,
                    timestamp=ts,
                    current_value=corr_res.current_value,
                    limit_value=corr_res.limit_value,
                    estimated_risk_amount=risk_estimate.total_risk if risk_estimate else None,
                )

        # 9. Pre-Trade Margin Check (Task F6)
        required_margin: float | None = None
        if (
            self.config.enforce_margin_check
            or self.config.margin_call_level_pct is not None
            or expected_entry_price > 0.0
        ):
            try:
                snap_for_order = self.margin_model.compute_snapshot(
                    equity=current_equity,
                    active_positions=active_positions,
                    pending_entry_orders=pending_orders,
                    mark_prices=mark_prices,
                    instrument_by_symbol=inst_map,
                    default_instrument=inst,
                    account_currency=account_ccy,
                    fx_rates=fx_rates,
                    exclude_pending_symbol=working_intent.symbol,
                )
                self.last_margin_snapshot = snap_for_order
                passed_margin, req_margin, margin_reject = (
                    self.margin_model.check_pre_trade_margin(
                        intent=working_intent,
                        quantity=qty,
                        expected_entry_price=expected_entry_price,
                        snapshot=snap_for_order,
                        instrument_config=inst,
                        active_position=active_positions.get(working_intent.symbol),
                        account_currency=account_ccy,
                        fx_rates=fx_rates,
                    )
                )
                required_margin = req_margin
                if not passed_margin:
                    return self._build_rejection(
                        intent=working_intent,
                        reject_code=margin_reject or "INSUFFICIENT_FREE_MARGIN",
                        check_name="MARGIN_CHECK",
                        message=(
                            f"Required margin {req_margin:.2f} exceeds effective free margin "
                            f"{snap_for_order.effective_free_margin:.2f}."
                        ),
                        checks=checks,
                        timestamp=ts,
                        current_value=req_margin,
                        limit_value=snap_for_order.effective_free_margin,
                        estimated_risk_amount=risk_estimate.total_risk if risk_estimate else None,
                        required_margin=req_margin,
                    )
                checks.append(
                    RiskCheckResult(
                        check_name="MARGIN_CHECK",
                        passed=True,
                        current_value=req_margin,
                        limit_value=snap_for_order.effective_free_margin,
                    )
                )
            except ValueError as exc:
                if self.config.enforce_margin_check:
                    return self._build_rejection(
                        intent=working_intent,
                        reject_code=str(exc),
                        check_name="MARGIN_CHECK",
                        checks=checks,
                        timestamp=ts,
                    )

        decision = RiskDecision(
            status=RiskDecisionStatus.APPROVED,
            approved_intent=working_intent,
            reject_reason=None,
            checks=tuple(checks),
            sized_quantity=qty,
            estimated_risk_amount=risk_estimate.total_risk if risk_estimate else None,
            required_margin=required_margin,
        )
        return self._record_decision(decision, working_intent, timestamp=ts)

    def approve_order_intent(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        reference_price: float | None = None,
        context: RiskContext | None = None,
    ) -> OrderIntent | None:
        """
        Backward-compatible entry point returning OrderIntent | None.
        """
        decision = self.evaluate_order_intent(
            intent=intent,
            current_balance=current_balance,
            current_equity=current_equity,
            reference_price=reference_price,
            context=context,
        )
        return decision.approved_intent

    def finalize_entry_sl_tp(
        self,
        order: Any,
        proposed_fill_price: float,
        context: RiskContext | None = None,
    ) -> tuple[bool, str, float | None, float | None]:
        """
        Phase 2 of Two-Phase Fill (Propose -> Validate/Finalize -> Commit):
        Validates SL/TP at actual fill price and re-verifies Trading Halt,
        Per-Trade Risk Cap, and Margin at fill time when context is provided.
        """
        fill_p = float(proposed_fill_price)
        if not math.isfinite(fill_p) or fill_p <= 0.0:
            self.last_reject_reason = "INVALID_FILL_PRICE"
            return False, "INVALID_FILL_PRICE", None, None

        if self.halt_manager.is_halted:
            reason = self.halt_manager.primary_reject_code or "TRADING_HALTED"
            self.last_reject_reason = reason
            return False, reason, None, None

        sl_p = order.stop_loss_price
        tp_p = order.take_profit_price
        sl_pct = order.stop_loss_pct
        tp_pct = order.take_profit_pct

        if sl_p is None and sl_pct is None and self.config.stop_loss_pct > 0:
            sl_pct = self.config.stop_loss_pct
        if tp_p is None and tp_pct is None and self.config.take_profit_pct > 0:
            tp_pct = self.config.take_profit_pct

        if sl_p is None and sl_pct is not None and sl_pct > 0:
            sl_p = (
                fill_p * (1.0 - sl_pct)
                if order.side == OrderSide.BUY
                else fill_p * (1.0 + sl_pct)
            )

        if tp_p is None and tp_pct is not None and tp_pct > 0:
            tp_p = (
                fill_p * (1.0 + tp_pct)
                if order.side == OrderSide.BUY
                else fill_p * (1.0 - tp_pct)
            )

        if sl_p is not None:
            if order.side == OrderSide.BUY and sl_p >= fill_p:
                self.last_reject_reason = "INVALID_SL_AT_FILL_PRICE"
                return False, "INVALID_SL_AT_FILL_PRICE", None, None
            if order.side == OrderSide.SELL and sl_p <= fill_p:
                self.last_reject_reason = "INVALID_SL_AT_FILL_PRICE"
                return False, "INVALID_SL_AT_FILL_PRICE", None, None

        if tp_p is not None:
            if order.side == OrderSide.BUY and tp_p <= fill_p:
                self.last_reject_reason = "INVALID_TP_AT_FILL_PRICE"
                return False, "INVALID_TP_AT_FILL_PRICE", None, None
            if order.side == OrderSide.SELL and tp_p >= fill_p:
                self.last_reject_reason = "INVALID_TP_AT_FILL_PRICE"
                return False, "INVALID_TP_AT_FILL_PRICE", None, None

        # Re-check Per-Trade Risk Cap and Margin at fill price if context is provided
        if context is not None:
            inst = self._resolve_instrument(order.symbol, context)
            account_ccy = context.account_currency or self.account_currency
            max_allowed_risk = self._compute_max_allowed_trade_risk(
                current_balance=context.current_balance,
                current_equity=context.current_equity,
            )
            if max_allowed_risk is not None and sl_p is not None:
                temp_intent = OrderIntent(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    limit_price=fill_p,
                    stop_loss_price=sl_p,
                    take_profit_price=tp_p,
                )
                est, est_err = self.estimator.estimate_trade_risk(
                    intent=temp_intent,
                    quantity=order.quantity,
                    instrument_config=inst,
                    reference_price=fill_p,
                    quote_bar=None,
                    account_currency=account_ccy,
                    fx_rates=context.fx_rates,
                )
                if est_err is not None or est is None:
                    reason = est_err or "INVALID_SL_AT_FILL_PRICE"
                    self.last_reject_reason = reason
                    return False, reason, None, None
                if est.total_risk > max_allowed_risk + 1e-9:
                    self.last_reject_reason = "MAX_TRADE_RISK_EXCEEDED"
                    return False, "MAX_TRADE_RISK_EXCEEDED", None, None

            if self.config.enforce_margin_check:
                inst_map = {**self.instrument_by_symbol, **(context.instrument_by_symbol or {})}
                snap = self.margin_model.compute_snapshot(
                    equity=context.current_equity,
                    active_positions=context.active_positions,
                    pending_entry_orders=context.pending_entry_orders,
                    mark_prices=context.mark_prices,
                    instrument_by_symbol=inst_map,
                    default_instrument=inst,
                    account_currency=account_ccy,
                    fx_rates=context.fx_rates,
                    exclude_pending_symbol=order.symbol,
                )
                temp_intent = OrderIntent(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    limit_price=fill_p,
                )
                passed_m, _, m_reject = self.margin_model.check_pre_trade_margin(
                    intent=temp_intent,
                    quantity=order.quantity,
                    expected_entry_price=fill_p,
                    snapshot=snap,
                    instrument_config=inst,
                    active_position=context.active_positions.get(order.symbol),
                    account_currency=account_ccy,
                    fx_rates=context.fx_rates,
                )
                if not passed_m:
                    reason = m_reject or "INSUFFICIENT_FREE_MARGIN"
                    self.last_reject_reason = reason
                    return False, reason, None, None

        return True, "OK", sl_p, tp_p

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
        eff_sl_pct = (
            stop_loss_pct
            if (stop_loss_pct is not None or stop_loss_price is not None)
            else cfg.stop_loss_pct
        )
        eff_tp_pct = (
            take_profit_pct
            if (take_profit_pct is not None or take_profit_price is not None)
            else cfg.take_profit_pct
        )

        if side == OrderSide.BUY:
            sl_p = (
                stop_loss_price
                if stop_loss_price is not None
                else (entry_price * (1.0 - eff_sl_pct) if eff_sl_pct else None)
            )
            tp_p = (
                take_profit_price
                if take_profit_price is not None
                else (entry_price * (1.0 + eff_tp_pct) if eff_tp_pct else None)
            )
        else:
            sl_p = (
                stop_loss_price
                if stop_loss_price is not None
                else (entry_price * (1.0 + eff_sl_pct) if eff_sl_pct else None)
            )
            tp_p = (
                take_profit_price
                if take_profit_price is not None
                else (entry_price * (1.0 - eff_tp_pct) if eff_tp_pct else None)
            )
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

        if side == OrderSide.BUY:
            if sl_price is not None and open_price <= sl_price:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    limit_price=open_price,
                    reason="STOP_GAP",
                )
            if tp_price is not None and open_price >= tp_price:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    limit_price=open_price,
                    reason="TAKE_PROFIT_GAP",
                )

            sl_hit = sl_price is not None and low <= sl_price
            tp_hit = tp_price is not None and high >= tp_price

            if sl_hit:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    limit_price=sl_price,
                    reason="STOP_LOSS",
                )
            if tp_hit:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    limit_price=tp_price,
                    reason="TAKE_PROFIT",
                )

        elif side == OrderSide.SELL:
            if sl_price is not None and open_price >= sl_price:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    limit_price=open_price,
                    reason="STOP_GAP",
                )
            if tp_price is not None and open_price <= tp_price:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    limit_price=open_price,
                    reason="TAKE_PROFIT_GAP",
                )

            sl_hit = sl_price is not None and high >= sl_price
            tp_hit = tp_price is not None and low <= tp_price

            if sl_hit:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    limit_price=sl_price,
                    reason="STOP_LOSS",
                )
            if tp_hit:
                return OrderIntent(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    limit_price=tp_price,
                    reason="TAKE_PROFIT",
                )

        return None
