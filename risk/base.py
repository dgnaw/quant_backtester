from abc import ABC, abstractmethod
import pandas as pd
from quant_backtester.models.order import OrderIntent, OrderSide


class BaseRiskManager(ABC):
    last_reject_reason: str | None = None

    @abstractmethod
    def approve_order_intent(
        self,
        intent: OrderIntent,
        current_balance: float,
        current_equity: float,
        reference_price: float | None = None,
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
