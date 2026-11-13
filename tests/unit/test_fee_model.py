import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.execution.fee import (
    CompositeFeeModel,
    FixedCommissionModel,
    PercentageFeeModel,
    ZeroFeeModel,
)
from quant_backtester.models.order import OrderSide
from quant_backtester.strategies.base import Strategy, StrategySignals


def test_zero_and_percentage_fee_backward_compatible():
    zero = ZeroFeeModel()
    assert zero.calculate_fee(price=2000.0, units=1.0) == 0.0
    assert zero.calculate_fee(2000.0, 1.0, quantity=0.01, side=OrderSide.BUY) == 0.0

    pct = PercentageFeeModel(fee_rate_per_side=0.00015)
    assert pct.calculate_fee(price=2000.0, units=1.0) == pytest.approx(0.30)
    assert pct.calculate_fee(2000.0, 1.0, quantity=0.01, side=OrderSide.SELL, is_opening=False) == pytest.approx(0.30)


def test_fixed_commission_per_side_and_round_turn():
    # PER_SIDE: 7 USD / lot on each fill
    per_side = FixedCommissionModel(amount_per_lot=7.0, mode="PER_SIDE")
    assert per_side.calculate_fee(2000.0, 1.0, quantity=0.01) == pytest.approx(0.07)
    assert per_side.calculate_fee(2000.0, 7.0, quantity=0.07) == pytest.approx(0.49)
    assert per_side.calculate_fee(2000.0, 100.0, quantity=1.0) == pytest.approx(7.0)

    # ROUND_TURN: 7 USD / lot / round-turn -> 3.5 USD / lot per fill
    rt = FixedCommissionModel(amount_per_lot=7.0, mode="ROUND_TURN")
    open_fee = rt.calculate_fee(2000.0, 1.0, quantity=0.01, is_opening=True)
    close_fee = rt.calculate_fee(2010.0, 1.0, quantity=0.01, is_opening=False)
    assert open_fee == pytest.approx(0.035)
    assert close_fee == pytest.approx(0.035)
    assert open_fee + close_fee == pytest.approx(0.07)


def test_composite_fee_model_and_currency_validation():
    comp = CompositeFeeModel([
        PercentageFeeModel(fee_rate_per_side=0.00015),
        FixedCommissionModel(amount_per_lot=7.0, mode="ROUND_TURN"),
    ])
    # 0.30 + 0.035 = 0.335
    total = comp.calculate_fee(2000.0, 1.0, quantity=0.01)
    assert total == pytest.approx(0.335)

    # Unsupported currency conversion
    jpy_fee = FixedCommissionModel(amount_per_lot=700.0, mode="ROUND_TURN", currency="JPY")
    with pytest.raises(ValueError, match="CURRENCY"):
        jpy_fee.validate_currency("USD")


def test_engine_wires_commission_per_lot_from_config():
    class SingleTradeStrategy(Strategy):
        name = "single_trade"

        def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
            df = market_data.copy()
            df["entry_signal"] = 0
            df["exit_long_signal"] = False
            df["exit_short_signal"] = False
            df.loc[0, "entry_signal"] = 1
            df.loc[1, "exit_long_signal"] = True
            return StrategySignals(df)

    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=4, freq="15min"),
        "open": [2000.0, 2000.0, 2010.0, 2010.0],
        "high": [2005.0, 2005.0, 2015.0, 2015.0],
        "low": [1995.0, 1995.0, 2005.0, 2005.0],
        "close": [2000.0, 2005.0, 2010.0, 2010.0],
    })
    cfg = BacktestConfig(
        lot_size=0.01,
        contract_size=100.0,
        fee_rate_per_side=0.0,
        commission_per_lot=7.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
    )
    res = BacktestEngine(cfg).run(df, SingleTradeStrategy())
    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    # 7 USD/lot round-turn * 0.01 lot = 0.035 entry + 0.035 exit = 0.07 total commission
    assert trade["entry_fee"] + trade["exit_fee"] == pytest.approx(0.07)

