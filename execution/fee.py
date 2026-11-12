from abc import ABC, abstractmethod


class BaseFeeModel(ABC):
    @abstractmethod
    def calculate_fee(self, price: float, units: float) -> float:
        raise NotImplementedError


class PercentageFeeModel(BaseFeeModel):
    def __init__(self, fee_rate_per_side: float = 0.00015):
        self.fee_rate_per_side = float(fee_rate_per_side)

    def calculate_fee(self, price: float, units: float) -> float:
        return abs(float(price) * float(units)) * self.fee_rate_per_side

