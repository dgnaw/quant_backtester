import pytest
import pandas as pd
from quant_backtester.config import EngineConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.execution.base import ExecutionModel
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide


class AlwaysLongStrategy(Strategy):
    name = "always_long"

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = 1
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)


class AlwaysShortStrategy(Strategy):
    name = "always_short"

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = -1
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)


class MockZeroExecutionModel(ExecutionModel):
    """Execution model tùy biến: Khớp lệnh với 0 trượt giá và 0 phí."""
    def execute_order(self, order, bar):
        return Fill(
            fill_id="MOCK-001",
            order_id=order.order_id,
            time=pd.Timestamp(bar["time"]),
            symbol=order.symbol,
            side=order.side,
            price=float(bar["open"]),
            quantity=order.quantity,
            units=order.units,
            fee=0.0,
            slippage=0.0,
            reason="MOCK_EXECUTION",
        )


def test_engine_runs_different_strategies_without_modifications():
    df = pd.DataFrame([
        {"time": "2026-01-01 00:00", "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0},
        {"time": "2026-01-01 00:15", "open": 2005.0, "high": 2020.0, "low": 2000.0, "close": 2015.0},
        {"time": "2026-01-01 00:30", "open": 2015.0, "high": 2025.0, "low": 2010.0, "close": 2020.0},
    ])
    cfg = EngineConfig()
    engine = BacktestEngine(config=cfg)

    # Chạy Strategy 1 (Long)
    res_long = engine.run(df, AlwaysLongStrategy())
    assert res_long.strategy_name == "always_long"
    assert len(res_long.trades) > 0
    assert res_long.trades.iloc[0]["direction"] == "LONG"

    # Chạy Strategy 2 (Short) trên cùng một engine instance
    res_short = engine.run(df, AlwaysShortStrategy())
    assert res_short.strategy_name == "always_short"
    assert len(res_short.trades) > 0
    assert res_short.trades.iloc[0]["direction"] == "SHORT"


def test_engine_supports_custom_execution_model_injection():
    df = pd.DataFrame([
        {"time": "2026-01-01 00:00", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0},
        {"time": "2026-01-01 00:15", "open": 102.0, "high": 110.0, "low": 100.0, "close": 108.0},
    ])
    cfg = EngineConfig()
    custom_exec = MockZeroExecutionModel()
    engine = BacktestEngine(config=cfg, execution_model=custom_exec)

    res = engine.run(df, AlwaysLongStrategy())
    assert len(res.trades) > 0
    # Đảm bảo fill fee được tính bằng 0 theo đúng mock execution model
    assert res.trades.iloc[0]["entry_fee"] == 0.0

