from abc import ABC, abstractmethod
from quant_backtester.models.order import OrderSide


class BaseSlippageModel(ABC):
    @abstractmethod
    def calculate_fill_price(self, price: float, side: OrderSide) -> float:
        raise NotImplementedError


class FixedSlippageModel(BaseSlippageModel):
    def __init__(self, slippage_price: float = 0.0):
        self.slippage_price = float(slippage_price)

    def calculate_fill_price(self, price: float, side: OrderSide) -> float:
        p = float(price)
        if side == OrderSide.BUY:
            return p + self.slippage_price
        else:
            return p - self.slippage_price

