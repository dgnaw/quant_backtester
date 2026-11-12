from abc import ABC, abstractmethod
import pandas as pd
from quant_backtester.models.order import Order
from quant_backtester.models.fill import Fill


class ExecutionModel(ABC):
    @abstractmethod
    def execute_order(self, order: Order, bar: pd.Series) -> Fill | None:
        raise NotImplementedError

