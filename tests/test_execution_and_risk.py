import pytest
import pandas as pd
from quant_backtester.execution.fee import PercentageFeeModel
from quant_backtester.execution.slippage import FixedSlippageModel
from quant_backtester.execution.immediate import ImmediateExecutionModel
from quant_backtester.execution.order_manager import OrderManager
from quant_backtester.risk.simple import SimpleRiskManager
from quant_backtester.config import RiskConfig
from quant_backtester.models.order import OrderIntent, OrderSide, OrderType, OrderStatus


def test_fee_model():
    fee_model = PercentageFeeModel(fee_rate_per_side=0.00015)
    # 2000.0 * 1.0 unit * 0.00015 = 0.30
    assert fee_model.calculate_fee(price=2000.0, units=1.0) == pytest.approx(0.30)


def test_slippage_model():
    slip_model = FixedSlippageModel(slippage_price=0.5)
    # Buy fill: 2000.0 + 0.5 = 2000.5
    assert slip_model.calculate_fill_price(price=2000.0, side=OrderSide.BUY) == 2000.5
    # Sell fill: 2000.0 - 0.5 = 1999.5
    assert slip_model.calculate_fill_price(price=2000.0, side=OrderSide.SELL) == 1999.5


def test_order_manager_and_execution_lifecycle():
    om = OrderManager()
    order = om.create_order(
        time=pd.Timestamp("2026-01-01 00:00"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        units=1.0,
    )
    assert order.status == OrderStatus.PENDING

    exec_model = ImmediateExecutionModel(
        fee_model=PercentageFeeModel(0.00015),
        slippage_model=FixedSlippageModel(0.2),
    )
    bar = pd.Series({"time": pd.Timestamp("2026-01-01 00:15"), "open": 2000.0, "high": 2010.0, "low": 1995.0, "close": 2005.0})
    fill = exec_model.execute_order(order, bar)

    assert fill is not None
    assert fill.price == 2000.2  # 2000.0 + 0.2 slippage
    assert fill.fee == pytest.approx(2000.2 * 1.0 * 0.00015)
    assert order.status == OrderStatus.FILLED


def test_simple_risk_manager_sl_tp():
    risk_mgr = SimpleRiskManager(config=RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02))

    # Đang giữ Long tại giá 2000.0
    # SL = 1980.0, TP = 2040.0
    bar_normal = pd.Series({"time": pd.Timestamp("2026-01-01 00:15"), "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0})
    exit_intent = risk_mgr.check_intrabar_exit(symbol="XAUUSD", side=OrderSide.BUY, entry_price=2000.0, units=1.0, bar=bar_normal)
    assert exit_intent is None

    # Bar có Low chạm 1975.0 <= SL -> Trigger Stop-Loss
    bar_sl = pd.Series({"time": pd.Timestamp("2026-01-01 00:30"), "open": 1990.0, "high": 1995.0, "low": 1975.0, "close": 1980.0})
    exit_intent_sl = risk_mgr.check_intrabar_exit(symbol="XAUUSD", side=OrderSide.BUY, entry_price=2000.0, units=1.0, bar=bar_sl)
    assert exit_intent_sl is not None
    assert exit_intent_sl.side == OrderSide.SELL
    assert exit_intent_sl.reason == "STOP_LOSS"
    assert exit_intent_sl.limit_price == pytest.approx(1980.0)  # Khớp tại mức SL

