import pytest
import pandas as pd
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.registry import StrategyRegistry


class DummyStrategyA(Strategy):
    name = "dummy_a"

    def __init__(self, param_x: int = 10):
        self.param_x = param_x

    def generate_signals(self, market_data):
        df = market_data.copy()
        df["entry_signal"] = 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)

    def get_parameters(self):
        return {"param_x": self.param_x}


class DummyStrategyB(Strategy):
    name = "dummy_b"

    def generate_signals(self, market_data):
        return StrategySignals(market_data)


def test_strategy_registration_and_lookup():
    StrategyRegistry.register("strategy_a", DummyStrategyA)
    StrategyRegistry.register("strategy_b", DummyStrategyB)

    assert "strategy_a" in StrategyRegistry.list_strategies()
    assert "strategy_b" in StrategyRegistry.list_strategies()

    # Khởi tạo qua get()
    instance_a = StrategyRegistry.get("strategy_a", param_x=42)
    assert isinstance(instance_a, DummyStrategyA)
    assert instance_a.param_x == 42
    assert instance_a.get_parameters() == {"param_x": 42}


def test_registry_raises_on_unregistered():
    with pytest.raises(KeyError, match="không tồn tại"):
        StrategyRegistry.get("non_existent_strategy")

