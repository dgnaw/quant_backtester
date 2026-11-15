"""
Base Risk Manager contract and RiskContext (Groups C, D, F).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_backtester.config import InstrumentConfig
from quant_backtester.models.order import Order, OrderIntent, OrderSide

if TYPE_CHECKING:
    from quant_backtester.execution.spread import QuoteBar
    from quant_backtester.portfolio.position import Position
    from quant_backtester.risk.decision import RiskDecision


@dataclass
class RiskContext:
    """
    Contextual portfolio, market, and account state passed to RiskManager
    for full Group F evaluation.
    """

    timestamp: pd.Timestamp | None = None
    bar_index: int = 0
    current_balance: float = 0.0
    current_equity: float = 0.0
    net_deposits: float = 0.0
    reference_price: float | None = None
    quote_bar: QuoteBar | None = None
    active_positions: dict[str, Position] = field(default_factory=dict)
    pending_entry_orders: list[Order] = field(default_factory=list)
    mark_prices: dict[str, float] = field(default_factory=dict)
    fx_rates: dict[str, float] = field(default_factory=dict)
    instrument_by_symbol: dict[str, InstrumentConfig] = field(default_factory=dict)
    account_currency: str = "USD"


class BaseRiskManager(ABC):
    """Abstract base class for Risk Managers."""

    last_reject_reason: str | None = None
    last_decision: RiskDecision | None = None

    @abstractmethod
    def approve_order_intent(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        reference_price: float | None = None,
        context: RiskContext | None = None,
    ) -> OrderIntent | None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
