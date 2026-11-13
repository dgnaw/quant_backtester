import pytest

from quant_backtester.execution.slippage import (
    FixedSlippageModel,
    PercentageSlippageModel,
    ZeroSlippageModel,
)
from quant_backtester.models.order import OrderSide


def test_zero_fixed_and_percentage_slippage():
    zero = ZeroSlippageModel()
    assert zero.calculate_fill_price(2000.0, OrderSide.BUY) == pytest.approx(2000.0)
    assert zero.calculate_fill_price(2000.0, OrderSide.SELL) == pytest.approx(2000.0)

    fixed = FixedSlippageModel(slippage_price=0.50)
    # BUY pays higher, SELL receives lower
    assert fixed.calculate_fill_price(2000.0, OrderSide.BUY) == pytest.approx(2000.50)
    assert fixed.calculate_fill_price(2000.0, OrderSide.SELL) == pytest.approx(1999.50)

    pct = PercentageSlippageModel(slippage_rate=0.001)  # 0.1% of 2000 = 2.0
    assert pct.calculate_fill_price(2000.0, OrderSide.BUY) == pytest.approx(2002.0)
    assert pct.calculate_fill_price(2000.0, OrderSide.SELL) == pytest.approx(1998.0)


def test_negative_or_invalid_slippage_rejected():
    with pytest.raises(ValueError):
        FixedSlippageModel(slippage_price=-0.1)
    with pytest.raises(ValueError):
        PercentageSlippageModel(slippage_rate=-0.001)

