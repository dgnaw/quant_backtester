import ast
from pathlib import Path
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode
from quant_backtester.strategies.context import StrategyContext


def test_strategies_package_dependency_direction():
    """
    C8 Dependency Direction:
    Các file trong strategies/ không được import trực tiếp implementation cụ thể của:
    Engine (BacktestEngine), Account, Portfolio, Ledger, hoặc ExecutionModel.
    """
    strategies_dir = Path(__file__).resolve().parents[2] / "strategies"
    forbidden_modules = {
        "engine",
        "quant_backtester.engine",
        "account",
        "quant_backtester.portfolio.account",
        "portfolio",
        "quant_backtester.portfolio.portfolio",
        "ledger",
        "quant_backtester.portfolio.ledger",
        "quant_backtester.execution",
        "quant_backtester.execution.base",
        "quant_backtester.execution.immediate",
    }
    forbidden_names = {"BacktestEngine", "Account", "Portfolio", "Ledger", "ExecutionModel", "ImmediateExecutionModel"}

    for py_file in strategies_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_modules, f"{py_file.name} import module bị cấm: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod not in forbidden_modules, f"{py_file.name} import từ module bị cấm: {mod}"
                for alias in node.names:
                    assert alias.name not in forbidden_names, f"{py_file.name} import class bị cấm: {alias.name}"


class MaliciousStrategy(Strategy):
    name = "malicious_strategy"
    mode = StrategyMode.EVENT_DRIVEN

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        # 1. Context must not expose account, portfolio, ledger, balance, or equity
        for forbidden_attr in ("account", "portfolio", "ledger", "balance", "equity", "engine"):
            assert not hasattr(context, forbidden_attr), f"Context không được lộ {forbidden_attr}"

        # 2. Attempting to mutate PositionView must fail
        with pytest.raises(Exception):
            context.position.units = 999.0  # type: ignore
        with pytest.raises(Exception):
            context.position.entry_price = 1.0  # type: ignore

        # 3. Attempting to mutate bar or history must not affect Engine
        bar["close"] = -999999.0
        hist = context.history(1)
        hist.loc[0, "close"] = -888888.0

        if context.bar_index == 0:
            return Signal(time=context.time, symbol=context.symbol, signal_type=SignalType.LONG)
        return None


def test_strategy_cannot_mutate_account_portfolio_or_market_data():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=4, freq="15min"),
        "open": [2000.0, 2010.0, 2020.0, 2030.0],
        "high": [2005.0, 2015.0, 2025.0, 2035.0],
        "low": [1995.0, 2005.0, 2015.0, 2025.0],
        "close": [2002.0, 2012.0, 2022.0, 2032.0],
    })
    df_orig = df.copy(deep=True)
    cfg = BacktestConfig(initial_balance=1000.0, stop_loss_pct=0.0, take_profit_pct=0.0, fee_rate_per_side=0.0)
    res = BacktestEngine(cfg).run(df, MaliciousStrategy())

    pd.testing.assert_frame_equal(df, df_orig)
    assert len(res.trades) == 1
    # Entry at bar 1 open = 2010.0, force close at last bar close = 2032.0 (not -999999.0)
    assert res.trades.iloc[0]["entry_price"] == pytest.approx(2010.0)
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(2032.0)

