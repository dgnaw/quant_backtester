from __future__ import annotations
from abc import ABC, abstractmethod
import math

from quant_backtester.models.order import OrderSide


class BaseSlippageModel(ABC):
    @abstractmethod
    def calculate_fill_price(self, price: float, side: OrderSide) -> float:
        raise NotImplementedError


class ZeroSlippageModel(BaseSlippageModel):
    def calculate_fill_price(self, price: float, side: OrderSide) -> float:
        return float(price)


class FixedSlippageModel(BaseSlippageModel):
    def __init__(self, slippage_price: float = 0.0):
        s = float(slippage_price)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError("slippage_price phải hữu hạn và >= 0.")
        self.slippage_price = s

    def calculate_fill_price(self, price: float, side: OrderSide) -> float:
        p = float(price)
        if side == OrderSide.BUY:
            return p + self.slippage_price
        return p - self.slippage_price


class PercentageSlippageModel(BaseSlippageModel):
    def __init__(self, slippage_rate: float = 0.0):
        r = float(slippage_rate)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError("slippage_rate phải hữu hạn và >= 0.")
        self.slippage_rate = r

    def calculate_fill_price(self, price: float, side: OrderSide) -> float:
        p = float(price)
        slip = abs(p) * self.slippage_rate
        if side == OrderSide.BUY:
            return p + slip
        return p - slip
