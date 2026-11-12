from typing import Any, ClassVar
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.parameters import Parameter


class StatefulCounterStrategy(Strategy):
    name = "stateful_counter"
    parameters: ClassVar[tuple[Parameter, ...]] = (
        Parameter(name="trigger_bar", param_type=int, default=2, min_value=0),
    )

    def _initial_state(self) -> dict[str, Any]:
        return {"bars_seen": 0, "fills_seen": 0, "rejections_seen": 0, "history_log": []}

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        self.state["bars_seen"] += len(df)
        self.state["history_log"].append(len(df))
        df["entry_signal"] = 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        if len(df) > self.trigger_bar:
            df.loc[self.trigger_bar, "entry_signal"] = 1
        return StrategySignals(df)

    def on_fill(self, fill_view: Any) -> None:
        self.state["fills_seen"] += 1
        # Thử can thiệp vào fill_view -> phải bị chặn vì chỉ đọc
        with pytest.raises((AttributeError, TypeError)):
            fill_view.price = 999999.0

    def on_intent_rejected(self, intent: Any, reason: str) -> None:
        self.state["rejections_seen"] += 1


def _make_df(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=n, freq="15min"),
        "open": [100.0 + i for i in range(n)],
        "high": [105.0 + i for i in range(n)],
        "low": [95.0 + i for i in range(n)],
        "close": [101.0 + i for i in range(n)],
    })


def test_state_initialization_and_reset_preserves_parameters():
    strat = StatefulCounterStrategy(trigger_bar=3)
    assert strat.state == {"bars_seen": 0, "fills_seen": 0, "rejections_seen": 0, "history_log": []}

    strat.state["bars_seen"] = 42
    strat.state["history_log"].append(99)
    strat.reset()

    assert strat.state == {"bars_seen": 0, "fills_seen": 0, "rejections_seen": 0, "history_log": []}
    assert strat.get_parameters() == {"trigger_bar": 3}


def test_two_instances_do_not_share_state():
    s1 = StatefulCounterStrategy(trigger_bar=1)
    s2 = StatefulCounterStrategy(trigger_bar=2)

    s1.state["bars_seen"] = 10
    s1.state["history_log"].append(10)

    assert s2.state["bars_seen"] == 0
    assert s2.state["history_log"] == []


def test_engine_resets_strategy_before_each_run():
    df = _make_df(8)
    strat = StatefulCounterStrategy(trigger_bar=1)
    engine = BacktestEngine(BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0))

    res1 = engine.run(df, strat)
    assert strat.state["bars_seen"] == 8
    assert strat.state["fills_seen"] >= 1

    res2 = engine.run(df, strat)
    # Vì Engine gọi strategy.reset() ở đầu mỗi run(), bars_seen sau run thứ 2 vẫn là 8 chứ không phải 16
    assert strat.state["bars_seen"] == 8
    assert res1.final_balance == pytest.approx(res2.final_balance)
    assert len(res1.trades) == len(res2.trades)

