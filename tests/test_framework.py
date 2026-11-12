import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.data import validate_ohlc
from quant_backtester.engine import BacktestEngine
from quant_backtester.strategies.base import Strategy, StrategySignals


class OneLongTrade(Strategy):
    name = "one_long_trade"

    def generate_signals(self, market_data):
        df = market_data.copy()
        df["entry_signal"] = 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        df.loc[0, "entry_signal"] = 1
        df.loc[2, "exit_long_signal"] = True
        return StrategySignals(df)


def market(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


def test_entry_executes_at_next_bar_open_and_exit_next_open():
    df = market([
        ["2026-01-01 00:00", 100, 101, 99, 100],
        ["2026-01-01 00:15", 102, 104, 101, 103],
        ["2026-01-01 00:30", 104, 105, 103, 104],
        ["2026-01-01 00:45", 106, 107, 105, 106],
    ])
    cfg = BacktestConfig(initial_balance=1000, lot_size=1, contract_size=1,
                         fee_rate_per_side=0, slippage_price=0,
                         stop_loss_pct=0, take_profit_pct=0)
    result = BacktestEngine(cfg).run(df, OneLongTrade())
    trade = result["trades"].iloc[0]
    assert trade["entry_price"] == 102  # signal at prior candle close
    assert trade["exit_price"] == 106  # exit signal at prior candle close
    assert trade["gross_pnl"] == 4
    assert trade["net_pnl"] == 4


def test_fee_is_charged_on_entry_and_exit():
    df = market([
        ["2026-01-01 00:00", 100, 101, 99, 100],
        ["2026-01-01 00:15", 100, 101, 99, 100],
        ["2026-01-01 00:30", 110, 111, 109, 110],
    ])
    cfg = BacktestConfig(initial_balance=1000, lot_size=1, contract_size=1,
                         fee_rate_per_side=0.01, slippage_price=0,
                         stop_loss_pct=0, take_profit_pct=0)
    result = BacktestEngine(cfg).run(df, OneLongTrade())
    trade = result["trades"].iloc[0]
    assert trade["entry_fee"] == pytest.approx(1.0)
    assert trade["exit_fee"] == pytest.approx(1.1)
    assert trade["net_pnl"] == pytest.approx(7.9)


def test_ohlc_validation_rejects_invalid_candle():
    df = market([["2026-01-01", 100, 98, 99, 100]])
    with pytest.raises(ValueError):
        validate_ohlc(df)


def test_engine_returns_finite_equity_curve():
    n = 80
    close = 2000 + np.sin(np.arange(n) / 4) * 10
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "open": open_, "high": np.maximum(open_, close) + 1,
        "low": np.minimum(open_, close) - 1, "close": close,
    })
    from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
    result = BacktestEngine(BacktestConfig()).run(df, MeanReversionStrategy(window=10, std_multiplier=1.0))
    assert len(result["equity"]) == n
    assert np.isfinite(result["equity"]["equity"]).all()
