from __future__ import annotations
import pandas as pd
from quant_backtester.models.order import Order, OrderSide, OrderType, OrderStatus
from quant_backtester.models.fill import Fill
from .base import ExecutionModel
from .fee import BaseFeeModel, PercentageFeeModel
from .slippage import BaseSlippageModel, FixedSlippageModel


class ImmediateExecutionModel(ExecutionModel):
    def __init__(
        self,
        fee_model: BaseFeeModel | None = None,
        slippage_model: BaseSlippageModel | None = None,
    ):
        self.fee_model = fee_model if fee_model is not None else PercentageFeeModel()
        self.slippage_model = slippage_model if slippage_model is not None else FixedSlippageModel()
        self._fill_counter = 0

    def execute_order(self, order: Order, bar: pd.Series) -> Fill | None:
        if order.status != OrderStatus.PENDING:
            return None

        # Khớp lệnh Market theo giá Open của bar (hoặc giá limit chỉ định trước nếu có)
        raw_price = float(order.limit_price) if order.limit_price is not None else float(bar["open"])
        fill_price = self.slippage_model.calculate_fill_price(raw_price, order.side)
        fee = self.fee_model.calculate_fee(fill_price, order.units)

        self._fill_counter += 1
        fill = Fill(
            fill_id=f"FILL-{self._fill_counter:05d}",
            order_id=order.order_id,
            time=pd.Timestamp(bar["time"]),
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            units=order.units,
            fee=fee,
            slippage=abs(fill_price - raw_price),
            reason=order.reason,
        )

        order.status = OrderStatus.FILLED
        return fill

