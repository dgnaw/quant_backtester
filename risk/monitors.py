"""
Portfolio Risk Monitors: Daily Loss, Max Drawdown, Exposure, and Correlation (Tasks F3, F4, F5, F7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quant_backtester.config import InstrumentConfig, MissingCorrelationPolicy, RiskConfig
from quant_backtester.data.trading_calendar import TradingCalendar
from quant_backtester.models.order import OrderOrigin, OrderSide, OrderStatus
from quant_backtester.portfolio.position import PositionSide
from quant_backtester.risk.decision import RiskCheckResult
from quant_backtester.risk.estimator import resolve_fx_rate
from quant_backtester.risk.halt import HaltReason, TradingHaltManager

if TYPE_CHECKING:
    from quant_backtester.models.order import Order, OrderIntent
    from quant_backtester.portfolio.position import Position


class DailyLossMonitor:
    """
    Monitors realized + unrealized daily loss against configured daily limits (Task F3).

    Supports:
    - Configurable trading timezone & TradingCalendar.session_id
    - Cash-flow adjustment (deposits/withdrawals during the day do not distort daily PnL)
    - Automatic reset of HALTED_DAILY_LOSS on new trading day transition
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.risk_config = risk_config
        self.calendar = calendar or TradingCalendar()

        self.current_session_id: str | None = None
        self.day_start_equity: float | None = None
        self.day_start_net_deposits: float = 0.0
        self.current_daily_pnl: float = 0.0
        self.current_daily_loss: float = 0.0
        self.current_daily_loss_pct: float = 0.0

    @property
    def is_enabled(self) -> bool:
        return (
            self.risk_config.max_daily_loss_pct is not None
            or self.risk_config.max_daily_loss_amount is not None
        )

    def _resolve_session_id(self, timestamp: pd.Timestamp) -> str:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_local = ts.tz_convert(self.risk_config.trading_timezone)
        return self.calendar.session_id(ts_local)

    def update(
        self,
        timestamp: pd.Timestamp | None,
        current_equity: float,
        net_deposits: float = 0.0,
        halt_manager: TradingHaltManager | None = None,
    ) -> RiskCheckResult:
        """
        Update daily loss state for the given timestamp and equity.
        If a new trading session starts, resets day_start_equity and clears HALTED_DAILY_LOSS.
        """
        if timestamp is not None:
            session_id = self._resolve_session_id(timestamp)
        else:
            session_id = self.current_session_id or "DEFAULT_SESSION"

        if self.current_session_id is None or session_id != self.current_session_id:
            self.current_session_id = session_id
            self.day_start_equity = float(current_equity)
            self.day_start_net_deposits = float(net_deposits)
            if halt_manager is not None and halt_manager.has_reason(HaltReason.HALTED_DAILY_LOSS):
                halt_manager.clear_halt(HaltReason.HALTED_DAILY_LOSS)

        assert self.day_start_equity is not None
        adjusted_day_start_equity = self.day_start_equity + (
            float(net_deposits) - self.day_start_net_deposits
        )
        self.current_daily_pnl = float(current_equity) - adjusted_day_start_equity
        self.current_daily_loss = max(0.0, -self.current_daily_pnl)
        if adjusted_day_start_equity > 0.0:
            self.current_daily_loss_pct = self.current_daily_loss / adjusted_day_start_equity
        else:
            self.current_daily_loss_pct = 0.0

        if not self.is_enabled:
            return RiskCheckResult(
                check_name="DAILY_LOSS_LIMIT",
                passed=True,
                current_value=self.current_daily_loss,
            )

        breached_amount = (
            self.risk_config.max_daily_loss_amount is not None
            and self.current_daily_loss >= self.risk_config.max_daily_loss_amount - 1e-9
        )
        breached_pct = (
            self.risk_config.max_daily_loss_pct is not None
            and self.current_daily_loss_pct >= self.risk_config.max_daily_loss_pct - 1e-9
        )

        if breached_amount or breached_pct:
            if halt_manager is not None:
                halt_manager.trigger_halt(
                    HaltReason.HALTED_DAILY_LOSS,
                    timestamp=timestamp,
                    details=(
                        f"daily_loss={self.current_daily_loss:.2f}, "
                        f"daily_loss_pct={self.current_daily_loss_pct:.4%}"
                    ),
                )
            return RiskCheckResult(
                check_name="DAILY_LOSS_LIMIT",
                passed=False,
                reject_code="DAILY_LOSS_LIMIT_REACHED",
                message=(
                    f"Daily loss limit reached: loss={self.current_daily_loss:.2f} "
                    f"({self.current_daily_loss_pct:.2%})"
                ),
                current_value=self.current_daily_loss if breached_amount else self.current_daily_loss_pct,
                limit_value=(
                    self.risk_config.max_daily_loss_amount
                    if breached_amount
                    else self.risk_config.max_daily_loss_pct
                ),
            )

        return RiskCheckResult(
            check_name="DAILY_LOSS_LIMIT",
            passed=True,
            current_value=self.current_daily_loss,
            limit_value=self.risk_config.max_daily_loss_amount or self.risk_config.max_daily_loss_pct,
        )


class DrawdownMonitor:
    """
    Monitors High Water Mark (HWM) Equity and Maximum Drawdown Circuit Breaker (Task F4).

    Supports:
    - Cash-flow adjustment for deposits/withdrawals
    - Persistent HALTED_MAX_DRAWDOWN state once triggered
    - Optional liquidation flag (`liquidate_on_max_drawdown`)
    """

    def __init__(self, risk_config: RiskConfig) -> None:
        self.risk_config = risk_config
        self.high_water_mark_equity: float | None = None
        self.last_net_deposits: float = 0.0
        self.current_drawdown_amount: float = 0.0
        self.current_drawdown_pct: float = 0.0
        self.liquidation_requested: bool = False
        self._liquidation_triggered_once: bool = False

    @property
    def is_enabled(self) -> bool:
        return (
            self.risk_config.max_drawdown_pct is not None
            or self.risk_config.max_drawdown_amount is not None
        )

    def update(
        self,
        timestamp: pd.Timestamp | None,
        current_equity: float,
        net_deposits: float = 0.0,
        halt_manager: TradingHaltManager | None = None,
    ) -> RiskCheckResult:
        """
        Update High Water Mark and drawdown metrics.
        Triggers HALTED_MAX_DRAWDOWN if limits are breached.
        """
        if self.high_water_mark_equity is None:
            self.high_water_mark_equity = float(current_equity)
            self.last_net_deposits = float(net_deposits)
        else:
            delta_deposits = float(net_deposits) - self.last_net_deposits
            self.high_water_mark_equity += delta_deposits
            self.last_net_deposits = float(net_deposits)
            if current_equity > self.high_water_mark_equity:
                self.high_water_mark_equity = float(current_equity)

        self.current_drawdown_amount = max(0.0, self.high_water_mark_equity - float(current_equity))
        if self.high_water_mark_equity > 0.0:
            self.current_drawdown_pct = self.current_drawdown_amount / self.high_water_mark_equity
        else:
            self.current_drawdown_pct = 0.0

        # Check if already persistently halted for max drawdown
        already_halted = (
            halt_manager is not None and halt_manager.has_reason(HaltReason.HALTED_MAX_DRAWDOWN)
        )

        if not self.is_enabled and not already_halted:
            return RiskCheckResult(
                check_name="MAX_DRAWDOWN_LIMIT",
                passed=True,
                current_value=self.current_drawdown_amount,
            )

        breached_amount = (
            self.risk_config.max_drawdown_amount is not None
            and self.current_drawdown_amount >= self.risk_config.max_drawdown_amount - 1e-9
        )
        breached_pct = (
            self.risk_config.max_drawdown_pct is not None
            and self.current_drawdown_pct >= self.risk_config.max_drawdown_pct - 1e-9
        )

        if breached_amount or breached_pct or already_halted:
            if halt_manager is not None and not already_halted:
                halt_manager.trigger_halt(
                    HaltReason.HALTED_MAX_DRAWDOWN,
                    timestamp=timestamp,
                    details=(
                        f"drawdown={self.current_drawdown_amount:.2f}, "
                        f"drawdown_pct={self.current_drawdown_pct:.4%}, "
                        f"hwm={self.high_water_mark_equity:.2f}"
                    ),
                )
            if self.risk_config.liquidate_on_max_drawdown and not self._liquidation_triggered_once:
                self.liquidation_requested = True
                self._liquidation_triggered_once = True

            return RiskCheckResult(
                check_name="MAX_DRAWDOWN_LIMIT",
                passed=False,
                reject_code="MAX_DRAWDOWN_LIMIT_REACHED",
                message=(
                    f"Max drawdown limit reached: dd={self.current_drawdown_amount:.2f} "
                    f"({self.current_drawdown_pct:.2%})"
                ),
                current_value=(
                    self.current_drawdown_amount if breached_amount else self.current_drawdown_pct
                ),
                limit_value=(
                    self.risk_config.max_drawdown_amount
                    if breached_amount
                    else self.risk_config.max_drawdown_pct
                ),
            )

        return RiskCheckResult(
            check_name="MAX_DRAWDOWN_LIMIT",
            passed=True,
            current_value=self.current_drawdown_amount,
            limit_value=self.risk_config.max_drawdown_amount or self.risk_config.max_drawdown_pct,
        )

    def consume_liquidation_request(self) -> bool:
        """Consume and clear the one-time liquidation request flag."""
        if self.liquidation_requested:
            self.liquidation_requested = False
            return True
        return False


def _compute_notional_account_ccy(
    units: float,
    price: float,
    instrument: InstrumentConfig,
    account_currency: str,
    fx_rates: dict[str, float] | None,
) -> float:
    """Compute notional exposure in account_currency, raising ValueError if FX missing."""
    if units <= 0.0 or price <= 0.0:
        return 0.0
    notional_quote = float(units) * float(price)
    conv_rate = resolve_fx_rate(
        from_currency=instrument.quote_currency,
        to_currency=account_currency,
        fx_rates=fx_rates,
        reference_price=price,
        symbol=instrument.symbol,
        base_currency=instrument.base_currency,
    )
    if conv_rate is None or conv_rate <= 0.0:
        raise ValueError("CURRENCY_CONVERSION_UNAVAILABLE")
    return notional_quote * conv_rate


class ExposureMonitor:
    """
    Symbol and Account Gross Exposure Monitor with Pending Entry Reservation (Task F5).

    Key semantics (DF6):
    - For target symbol S: `OrderManager` only keeps 1 active PENDING `STRATEGY_ENTRY`
      order per symbol (superseding any older one). Therefore, on symbol S, `proposed_order`
      replaces any existing pending entry order on S rather than adding to it.
    - On other symbols k != S: includes active positions + pending entry orders if
      `include_pending_orders_in_exposure == True`.
    - Handles Netting opposite-side orders: if `proposed_order` is opposite to an open
      position on symbol S, the post-fill exposure on symbol S is `abs(pos_lots - order_lots)`.
    """

    def __init__(self, risk_config: RiskConfig) -> None:
        self.risk_config = risk_config

    @property
    def is_enabled(self) -> bool:
        return any(
            x is not None
            for x in (
                self.risk_config.max_symbol_lots,
                self.risk_config.max_account_lots,
                self.risk_config.max_symbol_notional,
                self.risk_config.max_account_notional,
            )
        )

    def check_exposure(
        self,
        intent: OrderIntent,
        quantity: float,
        expected_entry_price: float,
        active_positions: dict[str, Position] | None = None,
        pending_entry_orders: list[Order] | None = None,
        mark_prices: dict[str, float] | None = None,
        instrument_by_symbol: dict[str, InstrumentConfig] | None = None,
        default_instrument: InstrumentConfig | None = None,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
    ) -> RiskCheckResult:
        """Evaluate symbol and account lot/notional limits."""
        if not self.is_enabled:
            return RiskCheckResult(check_name="EXPOSURE_LIMIT", passed=True)

        positions = active_positions or {}
        pending_orders = pending_entry_orders or []
        marks = mark_prices or {}
        inst_map = instrument_by_symbol or {}
        fallback_inst = default_instrument or InstrumentConfig(symbol=intent.symbol)

        target_sym = intent.symbol
        target_inst = inst_map.get(target_sym, fallback_inst)

        # Calculate post-trade symbol S lots & notional
        pos_s = positions.get(target_sym)
        if pos_s is not None and not pos_s.is_flat:
            is_same_side = (
                (pos_s.side == PositionSide.LONG and intent.side == OrderSide.BUY)
                or (pos_s.side == PositionSide.SHORT and intent.side == OrderSide.SELL)
            )
            if is_same_side:
                symbol_s_lots = pos_s.quantity + float(quantity)
            else:
                # Opposite side reduces or reverses exposure
                symbol_s_lots = abs(pos_s.quantity - float(quantity))
        else:
            symbol_s_lots = float(quantity)

        symbol_s_units = symbol_s_lots * target_inst.contract_size
        need_notional = (
            self.risk_config.max_symbol_notional is not None
            or self.risk_config.max_account_notional is not None
        )

        try:
            symbol_s_notional = (
                _compute_notional_account_ccy(
                    units=symbol_s_units,
                    price=expected_entry_price,
                    instrument=target_inst,
                    account_currency=account_currency,
                    fx_rates=fx_rates,
                )
                if need_notional
                else 0.0
            )
        except ValueError as exc:
            return RiskCheckResult(
                check_name="EXPOSURE_LIMIT",
                passed=False,
                reject_code=str(exc),
                message="Currency conversion unavailable for symbol notional calculation.",
            )

        # 1. Check Symbol Lot Limit
        if (
            self.risk_config.max_symbol_lots is not None
            and symbol_s_lots > self.risk_config.max_symbol_lots + 1e-9
        ):
            return RiskCheckResult(
                check_name="EXPOSURE_LIMIT",
                passed=False,
                reject_code="SYMBOL_LOT_LIMIT_EXCEEDED",
                message=(
                    f"Symbol {target_sym} projected lots {symbol_s_lots:.4f} exceeds "
                    f"max_symbol_lots {self.risk_config.max_symbol_lots:.4f}"
                ),
                current_value=symbol_s_lots,
                limit_value=self.risk_config.max_symbol_lots,
            )

        # 2. Check Symbol Notional Limit
        if (
            self.risk_config.max_symbol_notional is not None
            and symbol_s_notional > self.risk_config.max_symbol_notional + 1e-9
        ):
            return RiskCheckResult(
                check_name="EXPOSURE_LIMIT",
                passed=False,
                reject_code="SYMBOL_NOTIONAL_LIMIT_EXCEEDED",
                message=(
                    f"Symbol {target_sym} projected notional {symbol_s_notional:.2f} exceeds "
                    f"max_symbol_notional {self.risk_config.max_symbol_notional:.2f}"
                ),
                current_value=symbol_s_notional,
                limit_value=self.risk_config.max_symbol_notional,
            )

        # 3. Accumulate other symbols k != S for Account Gross Exposure
        account_lots = symbol_s_lots
        account_notional = symbol_s_notional

        for sym, pos in positions.items():
            if sym == target_sym or pos.is_flat:
                continue
            inst = inst_map.get(sym, fallback_inst)
            account_lots += pos.quantity
            if need_notional:
                m_price = marks.get(sym, pos.avg_entry_price)
                try:
                    account_notional += _compute_notional_account_ccy(
                        units=pos.units,
                        price=m_price,
                        instrument=inst,
                        account_currency=account_currency,
                        fx_rates=fx_rates,
                    )
                except ValueError as exc:
                    return RiskCheckResult(
                        check_name="EXPOSURE_LIMIT",
                        passed=False,
                        reject_code=str(exc),
                        message=f"Currency conversion unavailable for open position {sym}.",
                    )

        if self.risk_config.include_pending_orders_in_exposure:
            for order in pending_orders:
                if order.status != OrderStatus.PENDING or order.origin != OrderOrigin.STRATEGY_ENTRY:
                    continue
                if order.symbol == target_sym:
                    # Replaced/superseded by proposed_order on target_sym
                    continue
                inst = inst_map.get(order.symbol, fallback_inst)
                account_lots += order.quantity
                if need_notional:
                    o_price = (
                        order.price
                        if (order.price is not None and order.price > 0.0)
                        else marks.get(order.symbol, 0.0)
                    )
                    try:
                        account_notional += _compute_notional_account_ccy(
                            units=order.quantity * inst.contract_size,
                            price=o_price,
                            instrument=inst,
                            account_currency=account_currency,
                            fx_rates=fx_rates,
                        )
                    except ValueError as exc:
                        return RiskCheckResult(
                            check_name="EXPOSURE_LIMIT",
                            passed=False,
                            reject_code=str(exc),
                            message=f"Currency conversion unavailable for pending order {order.symbol}.",
                        )

        # 4. Check Account Lot Limit
        if (
            self.risk_config.max_account_lots is not None
            and account_lots > self.risk_config.max_account_lots + 1e-9
        ):
            return RiskCheckResult(
                check_name="EXPOSURE_LIMIT",
                passed=False,
                reject_code="ACCOUNT_LOT_LIMIT_EXCEEDED",
                message=(
                    f"Account projected lots {account_lots:.4f} exceeds "
                    f"max_account_lots {self.risk_config.max_account_lots:.4f}"
                ),
                current_value=account_lots,
                limit_value=self.risk_config.max_account_lots,
            )

        # 5. Check Account Notional Limit
        if (
            self.risk_config.max_account_notional is not None
            and account_notional > self.risk_config.max_account_notional + 1e-9
        ):
            return RiskCheckResult(
                check_name="EXPOSURE_LIMIT",
                passed=False,
                reject_code="ACCOUNT_NOTIONAL_LIMIT_EXCEEDED",
                message=(
                    f"Account projected notional {account_notional:.2f} exceeds "
                    f"max_account_notional {self.risk_config.max_account_notional:.2f}"
                ),
                current_value=account_notional,
                limit_value=self.risk_config.max_account_notional,
            )

        return RiskCheckResult(
            check_name="EXPOSURE_LIMIT",
            passed=True,
            current_value=account_notional if need_notional else account_lots,
        )


class CorrelationRiskMonitor:
    """
    Pairwise Directional Correlation Risk Monitor (Task F7 - DF8).

    Evaluates effective directional correlation between the proposed order on symbol S
    and existing active positions (plus pending entry orders on k != S if enabled):
      effective_corr = rho(sym_a, sym_b) * direction_sign
    where direction_sign is +1 if same side (both LONG or both SHORT) and -1 if opposite.
    """

    def __init__(self, risk_config: RiskConfig) -> None:
        self.risk_config = risk_config

    @property
    def is_enabled(self) -> bool:
        return self.risk_config.enable_correlation_check and (
            self.risk_config.max_correlated_positions is not None
            or self.risk_config.max_correlated_notional is not None
        )

    def _lookup_rho(self, sym_a: str, sym_b: str) -> float | None:
        if sym_a == sym_b:
            return 1.0
        matrix = self.risk_config.correlation_matrix
        if not matrix:
            return None
        if (sym_a, sym_b) in matrix:
            return float(matrix[(sym_a, sym_b)])
        if (sym_b, sym_a) in matrix:
            return float(matrix[(sym_b, sym_a)])
        if sym_a in matrix and isinstance(matrix[sym_a], dict) and sym_b in matrix[sym_a]:
            return float(matrix[sym_a][sym_b])
        if sym_b in matrix and isinstance(matrix[sym_b], dict) and sym_a in matrix[sym_b]:
            return float(matrix[sym_b][sym_a])
        return None

    def check_correlation(
        self,
        intent: OrderIntent,
        quantity: float,
        expected_entry_price: float,
        active_positions: dict[str, Position] | None = None,
        pending_entry_orders: list[Order] | None = None,
        mark_prices: dict[str, float] | None = None,
        instrument_by_symbol: dict[str, InstrumentConfig] | None = None,
        default_instrument: InstrumentConfig | None = None,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
    ) -> RiskCheckResult:
        """Evaluate correlated position count and correlated notional exposure."""
        if not self.is_enabled:
            return RiskCheckResult(check_name="CORRELATION_RISK", passed=True)

        positions = active_positions or {}
        pending_orders = pending_entry_orders or []
        marks = mark_prices or {}
        inst_map = instrument_by_symbol or {}
        fallback_inst = default_instrument or InstrumentConfig(symbol=intent.symbol)

        target_sym = intent.symbol
        target_inst = inst_map.get(target_sym, fallback_inst)
        new_sign = 1 if intent.side == OrderSide.BUY else -1

        need_notional = self.risk_config.max_correlated_notional is not None
        try:
            new_notional = (
                _compute_notional_account_ccy(
                    units=float(quantity) * target_inst.contract_size,
                    price=expected_entry_price,
                    instrument=target_inst,
                    account_currency=account_currency,
                    fx_rates=fx_rates,
                )
                if need_notional
                else 0.0
            )
        except ValueError as exc:
            return RiskCheckResult(
                check_name="CORRELATION_RISK",
                passed=False,
                reject_code=str(exc),
                message="Currency conversion unavailable for correlation notional calculation.",
            )

        # Build list of other exposures: (symbol, sign, notional)
        # Note: For same symbol S, scale-in is governed by PositionMode & symbol exposure limits;
        # correlation risk checks cross-symbol or portfolio-wide correlated exposures.
        other_exposures: list[tuple[str, int, float]] = []

        for sym, pos in positions.items():
            if sym == target_sym or pos.is_flat:
                continue
            pos_sign = 1 if pos.side == PositionSide.LONG else -1
            inst = inst_map.get(sym, fallback_inst)
            m_price = marks.get(sym, pos.avg_entry_price)
            pos_notional = 0.0
            if need_notional:
                try:
                    pos_notional = _compute_notional_account_ccy(
                        units=pos.units,
                        price=m_price,
                        instrument=inst,
                        account_currency=account_currency,
                        fx_rates=fx_rates,
                    )
                except ValueError as exc:
                    return RiskCheckResult(
                        check_name="CORRELATION_RISK",
                        passed=False,
                        reject_code=str(exc),
                        message=f"Currency conversion unavailable for open position {sym}.",
                    )
            other_exposures.append((sym, pos_sign, pos_notional))

        if self.risk_config.include_pending_orders_in_exposure:
            for order in pending_orders:
                if order.status != OrderStatus.PENDING or order.origin != OrderOrigin.STRATEGY_ENTRY:
                    continue
                if order.symbol == target_sym:
                    continue
                ord_sign = 1 if order.side == OrderSide.BUY else -1
                inst = inst_map.get(order.symbol, fallback_inst)
                o_price = (
                    order.price
                    if (order.price is not None and order.price > 0.0)
                    else marks.get(order.symbol, 0.0)
                )
                ord_notional = 0.0
                if need_notional:
                    try:
                        ord_notional = _compute_notional_account_ccy(
                            units=order.quantity * inst.contract_size,
                            price=o_price,
                            instrument=inst,
                            account_currency=account_currency,
                            fx_rates=fx_rates,
                        )
                    except ValueError as exc:
                        return RiskCheckResult(
                            check_name="CORRELATION_RISK",
                            passed=False,
                            reject_code=str(exc),
                            message=f"Currency conversion unavailable for pending order {order.symbol}.",
                        )
                other_exposures.append((order.symbol, ord_sign, ord_notional))

        if not other_exposures:
            # Only the new position itself
            if (
                self.risk_config.max_correlated_positions is not None
                and 1 > self.risk_config.max_correlated_positions
            ):
                return RiskCheckResult(
                    check_name="CORRELATION_RISK",
                    passed=False,
                    reject_code="CORRELATED_POSITIONS_LIMIT_EXCEEDED",
                    current_value=1.0,
                    limit_value=float(self.risk_config.max_correlated_positions),
                )
            return RiskCheckResult(check_name="CORRELATION_RISK", passed=True)

        # Count correlated cluster including the proposed order itself (starts at 1)
        correlated_count = 1
        correlated_notional = new_notional
        threshold = self.risk_config.correlation_threshold
        policy = self.risk_config.missing_correlation_policy

        for other_sym, other_sign, other_notional in other_exposures:
            rho = self._lookup_rho(target_sym, other_sym)
            if rho is None:
                if policy == MissingCorrelationPolicy.REJECT:
                    return RiskCheckResult(
                        check_name="CORRELATION_RISK",
                        passed=False,
                        reject_code="MISSING_CORRELATION_DATA",
                        message=f"Missing correlation data between {target_sym} and {other_sym}.",
                    )
                elif policy == MissingCorrelationPolicy.IGNORE:
                    continue
                elif policy == MissingCorrelationPolicy.CONSERVATIVE_ONE:
                    effective_corr = 1.0
                else:
                    continue
            else:
                direction_sign = new_sign * other_sign
                effective_corr = rho * direction_sign

            if effective_corr >= threshold - 1e-9:
                correlated_count += 1
                correlated_notional += other_notional

        if (
            self.risk_config.max_correlated_positions is not None
            and correlated_count > self.risk_config.max_correlated_positions
        ):
            return RiskCheckResult(
                check_name="CORRELATION_RISK",
                passed=False,
                reject_code="CORRELATED_POSITIONS_LIMIT_EXCEEDED",
                message=(
                    f"Correlated positions count {correlated_count} exceeds "
                    f"max_correlated_positions {self.risk_config.max_correlated_positions}."
                ),
                current_value=float(correlated_count),
                limit_value=float(self.risk_config.max_correlated_positions),
            )

        if (
            self.risk_config.max_correlated_notional is not None
            and correlated_count > 1
            and correlated_notional > self.risk_config.max_correlated_notional + 1e-9
        ):
            return RiskCheckResult(
                check_name="CORRELATION_RISK",
                passed=False,
                reject_code="CORRELATED_EXPOSURE_LIMIT_EXCEEDED",
                message=(
                    f"Correlated notional {correlated_notional:.2f} exceeds "
                    f"max_correlated_notional {self.risk_config.max_correlated_notional:.2f}."
                ),
                current_value=correlated_notional,
                limit_value=self.risk_config.max_correlated_notional,
            )

        return RiskCheckResult(
            check_name="CORRELATION_RISK",
            passed=True,
            current_value=correlated_notional if need_notional else float(correlated_count),
        )
