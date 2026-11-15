"""
Margin Model, Margin Call & Stop-Out logic (Group F - Task F6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quant_backtester.config import InstrumentConfig, RiskConfig
from quant_backtester.models.order import OrderOrigin, OrderSide, OrderStatus
from quant_backtester.portfolio.position import PositionSide
from quant_backtester.risk.estimator import resolve_fx_rate

if TYPE_CHECKING:
    from quant_backtester.models.order import Order, OrderIntent
    from quant_backtester.portfolio.position import Position


@dataclass(frozen=True)
class MarginSnapshot:
    """Immutable snapshot of account margin state."""

    equity: float
    used_margin: float
    reserved_margin: float
    free_margin: float
    effective_free_margin: float
    margin_level_pct: float
    is_margin_call: bool = False
    is_stop_out: bool = False


class BaseMarginModel(ABC):
    """Abstract base class for margin models."""

    @abstractmethod
    def calculate_required_margin(
        self,
        units: float,
        price: float,
        instrument_config: InstrumentConfig,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
        leverage_override: float | None = None,
    ) -> float:
        """Compute required margin in account currency for given units and price."""


class LeverageMarginModel(BaseMarginModel):
    """
    Standard Leverage-based Margin Model (Task F6).

    Formula:
      notional_quote = units * price
      required_margin = (notional_quote * conversion_rate) / leverage
    """

    def __init__(
        self,
        risk_config: RiskConfig | None = None,
        default_leverage: float = 100.0,
    ) -> None:
        self.risk_config = risk_config or RiskConfig()
        self.default_leverage = (
            self.risk_config.leverage
            if self.risk_config.leverage is not None
            else default_leverage
        )

    def resolve_leverage(
        self,
        instrument_config: InstrumentConfig | None = None,
        leverage_override: float | None = None,
    ) -> float:
        """Resolve effective leverage for an instrument."""
        if leverage_override is not None:
            return float(leverage_override)
        if instrument_config is not None and instrument_config.leverage is not None:
            return float(instrument_config.leverage)
        if self.risk_config.leverage is not None:
            return float(self.risk_config.leverage)
        return float(self.default_leverage)

    def calculate_required_margin(
        self,
        units: float,
        price: float,
        instrument_config: InstrumentConfig,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
        leverage_override: float | None = None,
    ) -> float:
        """
        Compute required margin in account_currency.
        Raises ValueError("CURRENCY_CONVERSION_UNAVAILABLE") if conversion is required
        and no valid fx_rate is available.
        """
        if units <= 0.0 or price <= 0.0:
            return 0.0

        leverage = self.resolve_leverage(instrument_config, leverage_override)
        if leverage <= 0.0:
            raise ValueError(f"Invalid leverage {leverage}; must be > 0.")

        notional_quote = float(units) * float(price)
        conv_rate = resolve_fx_rate(
            from_currency=instrument_config.quote_currency,
            to_currency=account_currency,
            fx_rates=fx_rates,
            reference_price=price,
            symbol=instrument_config.symbol,
            base_currency=instrument_config.base_currency,
        )
        if conv_rate is None or conv_rate <= 0.0:
            raise ValueError("CURRENCY_CONVERSION_UNAVAILABLE")

        return (notional_quote * conv_rate) / leverage

    def compute_position_margin(
        self,
        position: Position,
        mark_price: float | None,
        instrument_config: InstrumentConfig,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
    ) -> float:
        """Compute used margin for a single active Position."""
        if position.is_flat or position.units <= 0.0:
            return 0.0
        price = (
            float(mark_price)
            if (mark_price is not None and mark_price > 0.0)
            else float(position.avg_entry_price)
        )
        return self.calculate_required_margin(
            units=position.units,
            price=price,
            instrument_config=instrument_config,
            account_currency=account_currency,
            fx_rates=fx_rates,
        )

    def compute_snapshot(
        self,
        equity: float,
        active_positions: dict[str, Position] | None = None,
        pending_entry_orders: list[Order] | None = None,
        mark_prices: dict[str, float] | None = None,
        instrument_by_symbol: dict[str, InstrumentConfig] | None = None,
        default_instrument: InstrumentConfig | None = None,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
        exclude_pending_symbol: str | None = None,
    ) -> MarginSnapshot:
        """
        Compute full MarginSnapshot across active positions and pending entry orders.

        Note on exclude_pending_symbol:
        When evaluating a new entry intent for symbol S, any existing PENDING
        STRATEGY_ENTRY order on symbol S will be superseded by OrderManager, so
        its reserved margin should be excluded to avoid double-counting.
        """
        positions = active_positions or {}
        marks = mark_prices or {}
        inst_map = instrument_by_symbol or {}
        fallback_inst = default_instrument or InstrumentConfig(symbol="DEFAULT")

        used_margin = 0.0
        for sym, pos in positions.items():
            if pos.is_flat:
                continue
            inst = inst_map.get(sym, fallback_inst)
            m_price = marks.get(sym, pos.avg_entry_price)
            used_margin += self.compute_position_margin(
                position=pos,
                mark_price=m_price,
                instrument_config=inst,
                account_currency=account_currency,
                fx_rates=fx_rates,
            )

        reserved_margin = 0.0
        if self.risk_config.reserve_margin_for_pending_orders and pending_entry_orders:
            for order in pending_entry_orders:
                if order.status != OrderStatus.PENDING or order.origin != OrderOrigin.STRATEGY_ENTRY:
                    continue
                if exclude_pending_symbol is not None and order.symbol == exclude_pending_symbol:
                    continue
                inst = inst_map.get(order.symbol, fallback_inst)
                order_units = order.quantity * inst.contract_size
                order_price = (
                    order.price
                    if (order.price is not None and order.price > 0.0)
                    else marks.get(order.symbol, 0.0)
                )
                if order_price > 0.0:
                    reserved_margin += self.calculate_required_margin(
                        units=order_units,
                        price=order_price,
                        instrument_config=inst,
                        account_currency=account_currency,
                        fx_rates=fx_rates,
                    )

        free_margin = float(equity) - used_margin
        effective_free_margin = float(equity) - used_margin - reserved_margin

        if used_margin <= 0.0:
            margin_level_pct = float("inf")
        else:
            margin_level_pct = (float(equity) / used_margin) * 100.0

        is_margin_call = False
        if (
            self.risk_config.margin_call_level_pct is not None
            and used_margin > 0.0
            and margin_level_pct <= self.risk_config.margin_call_level_pct
        ):
            is_margin_call = True

        is_stop_out = False
        if (
            self.risk_config.stop_out_level_pct is not None
            and used_margin > 0.0
            and margin_level_pct <= self.risk_config.stop_out_level_pct
        ):
            is_stop_out = True

        return MarginSnapshot(
            equity=float(equity),
            used_margin=used_margin,
            reserved_margin=reserved_margin,
            free_margin=free_margin,
            effective_free_margin=effective_free_margin,
            margin_level_pct=margin_level_pct,
            is_margin_call=is_margin_call,
            is_stop_out=is_stop_out,
        )

    def check_pre_trade_margin(
        self,
        intent: OrderIntent,
        quantity: float,
        expected_entry_price: float,
        snapshot: MarginSnapshot,
        instrument_config: InstrumentConfig,
        active_position: Position | None = None,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
    ) -> tuple[bool, float, str | None]:
        """
        Check whether sufficient free margin exists for the proposed order.

        Handles netting behavior:
        - If flat or same side as active_position: incremental units = new_units
        - If opposite side and new_units <= pos.units: reduces exposure -> incremental margin = 0
        - If opposite side and new_units > pos.units (over-close reverse):
          net post-trade margin is compared against current position margin.

        Returns (passed, additional_margin_required, reject_code).
        """
        new_units = float(quantity) * instrument_config.contract_size

        try:
            if active_position is not None and not active_position.is_flat:
                is_same_side = (
                    (active_position.side == PositionSide.LONG and intent.side == OrderSide.BUY)
                    or (active_position.side == PositionSide.SHORT and intent.side == OrderSide.SELL)
                )
                if not is_same_side:
                    if new_units <= active_position.units + 1e-12:
                        return True, 0.0, None
                    # Over-close reverse: net open after fill is (new_units - active_position.units)
                    net_post_units = new_units - active_position.units
                    post_margin = self.calculate_required_margin(
                        units=net_post_units,
                        price=expected_entry_price,
                        instrument_config=instrument_config,
                        account_currency=account_currency,
                        fx_rates=fx_rates,
                    )
                    current_pos_margin = self.compute_position_margin(
                        position=active_position,
                        mark_price=expected_entry_price,
                        instrument_config=instrument_config,
                        account_currency=account_currency,
                        fx_rates=fx_rates,
                    )
                    additional_margin = max(0.0, post_margin - current_pos_margin)
                else:
                    additional_margin = self.calculate_required_margin(
                        units=new_units,
                        price=expected_entry_price,
                        instrument_config=instrument_config,
                        account_currency=account_currency,
                        fx_rates=fx_rates,
                    )
            else:
                additional_margin = self.calculate_required_margin(
                    units=new_units,
                    price=expected_entry_price,
                    instrument_config=instrument_config,
                    account_currency=account_currency,
                    fx_rates=fx_rates,
                )
        except ValueError as exc:
            return False, 0.0, str(exc)

        if not self.risk_config.enforce_margin_check:
            return True, additional_margin, None

        if snapshot.is_margin_call:
            return False, additional_margin, "MARGIN_CALL_ACTIVE"

        if additional_margin > snapshot.effective_free_margin + 1e-9:
            return False, additional_margin, "INSUFFICIENT_FREE_MARGIN"

        return True, additional_margin, None

    def select_positions_for_stop_out(
        self,
        active_positions: dict[str, Position],
        mark_prices: dict[str, float] | None = None,
        instrument_by_symbol: dict[str, InstrumentConfig] | None = None,
        default_instrument: InstrumentConfig | None = None,
        account_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
    ) -> list[Position]:
        """
        Return open positions sorted in deterministic Stop-Out liquidation priority order:
        1. Largest unrealized loss first (`unrealized_pnl` ascending)
        2. Largest required margin first (`required_margin` descending)
        3. Symbol alphabetical (`symbol` A -> Z)
        """
        marks = mark_prices or {}
        inst_map = instrument_by_symbol or {}
        fallback_inst = default_instrument or InstrumentConfig(symbol="DEFAULT")

        candidates: list[tuple[float, float, str, Position]] = []
        for sym, pos in active_positions.items():
            if pos.is_flat:
                continue
            inst = inst_map.get(sym, fallback_inst)
            m_price = marks.get(sym, pos.avg_entry_price)
            try:
                req_margin = self.compute_position_margin(
                    position=pos,
                    mark_price=m_price,
                    instrument_config=inst,
                    account_currency=account_currency,
                    fx_rates=fx_rates,
                )
            except ValueError:
                req_margin = 0.0
            upnl = pos.unrealized_pnl(m_price) if callable(pos.unrealized_pnl) else float(pos.unrealized_pnl)
            candidates.append((float(upnl), -float(req_margin), sym, pos))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in candidates]
