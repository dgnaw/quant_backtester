from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import Iterable, Sequence

from quant_backtester.models.order import OrderSide


class BaseFeeModel(ABC):
    currency: str = "USD"

    def validate_currency(self, account_currency: str) -> None:
        if self.currency != account_currency:
            raise ValueError(
                f"UNSUPPORTED_CURRENCY_CONVERSION: Fee currency ({self.currency}) khác với Account currency ({account_currency})."
            )

    @abstractmethod
    def calculate_fee(
        self,
        price: float,
        units: float,
        *,
        quantity: float | None = None,
        side: OrderSide | None = None,
        is_opening: bool = True,
    ) -> float:
        raise NotImplementedError


class ZeroFeeModel(BaseFeeModel):
    def __init__(self, currency: str = "USD"):
        self.currency = currency

    def calculate_fee(
        self,
        price: float,
        units: float,
        *,
        quantity: float | None = None,
        side: OrderSide | None = None,
        is_opening: bool = True,
    ) -> float:
        return 0.0


class PercentageFeeModel(BaseFeeModel):
    def __init__(self, fee_rate_per_side: float = 0.00015, currency: str = "USD"):
        r = float(fee_rate_per_side)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError("fee_rate_per_side phải hữu hạn và >= 0.")
        self.fee_rate_per_side = r
        self.currency = currency

    def calculate_fee(
        self,
        price: float,
        units: float,
        *,
        quantity: float | None = None,
        side: OrderSide | None = None,
        is_opening: bool = True,
    ) -> float:
        return abs(float(price) * float(units)) * self.fee_rate_per_side


class FixedCommissionModel(BaseFeeModel):
    """
    Tính commission cố định theo số lot (quantity).
    Hỗ trợ 2 chế độ:
    - PER_SIDE: Fee = amount_per_lot * lots cho mỗi Fill.
    - ROUND_TURN: Fee per Fill = amount_per_lot * lots / 2.0 (mỗi chiều chịu một nửa).
    """

    def __init__(
        self,
        amount_per_lot: float = 0.0,
        mode: str = "ROUND_TURN",
        contract_size: float = 100.0,
        currency: str = "USD",
    ):
        amt = float(amount_per_lot)
        if not math.isfinite(amt) or amt < 0.0:
            raise ValueError("amount_per_lot phải hữu hạn và >= 0.")
        m = str(mode).strip().upper()
        if m not in ("PER_SIDE", "ROUND_TURN"):
            raise ValueError(f"Chế độ commission không hợp lệ: {mode!r}. Chỉ hỗ trợ PER_SIDE hoặc ROUND_TURN.")
        self.amount_per_lot = amt
        self.mode = m
        self.contract_size = float(contract_size) if contract_size > 0 else 100.0
        self.currency = currency

    def calculate_fee(
        self,
        price: float,
        units: float,
        *,
        quantity: float | None = None,
        side: OrderSide | None = None,
        is_opening: bool = True,
    ) -> float:
        lots = float(quantity) if quantity is not None else (abs(float(units)) / self.contract_size)
        if self.mode == "ROUND_TURN":
            return self.amount_per_lot * lots / 2.0
        return self.amount_per_lot * lots


class CompositeFeeModel(BaseFeeModel):
    def __init__(self, models: Sequence[BaseFeeModel] | Iterable[BaseFeeModel], currency: str = "USD"):
        self.models: list[BaseFeeModel] = list(models)
        self.currency = currency

    def validate_currency(self, account_currency: str) -> None:
        super().validate_currency(account_currency)
        for m in self.models:
            m.validate_currency(account_currency)

    def calculate_fee(
        self,
        price: float,
        units: float,
        *,
        quantity: float | None = None,
        side: OrderSide | None = None,
        is_opening: bool = True,
    ) -> float:
        return sum(
            m.calculate_fee(price, units, quantity=quantity, side=side, is_opening=is_opening)
            for m in self.models
        )
