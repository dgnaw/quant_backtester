import pytest
import pandas as pd
from datetime import datetime
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.models.order import Order, OrderType, OrderSide, OrderStatus, OrderIntent
from quant_backtester.models.fill import Fill
from quant_backtester.models.result import BacktestResult
from quant_backtester.config import EngineConfig


def test_signal_creation_and_validation():
    sig = Signal(time=pd.Timestamp("2026-01-01 00:00:00"), symbol="XAUUSD", signal_type=SignalType.LONG, strength=1.0)
    assert sig.signal_type == SignalType.LONG
    assert sig.is_entry is True
    assert sig.is_exit is False

    exit_sig = Signal(time=pd.Timestamp("2026-01-01 00:15:00"), symbol="XAUUSD", signal_type=SignalType.EXIT_LONG)
    assert exit_sig.is_entry is False
    assert exit_sig.is_exit is True


def test_order_intent_and_order():
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )
    assert intent.side == OrderSide.BUY

    order = Order(
        order_id="ORD-001",
        time=pd.Timestamp("2026-01-01 00:15:00"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        units=1.0,
    )
    assert order.status == OrderStatus.PENDING
    assert order.units == 1.0


def test_fill_creation():
    fill = Fill(
        fill_id="FILL-001",
        order_id="ORD-001",
        time=pd.Timestamp("2026-01-01 00:15:00"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        price=2000.0,
        quantity=0.01,
        units=1.0,
        fee=0.30,
        slippage=0.0,
    )
    assert fill.price == 2000.0
    assert fill.fee == 0.30


def test_backtest_result():
    equity_df = pd.DataFrame([{"time": "2026-01-01", "balance": 1000.0, "equity": 1000.0, "position": 0}])
    trades_df = pd.DataFrame([])
    orders_df = pd.DataFrame([])
    metrics = {"total_return_pct": 0.0, "sharpe_annual": 0.0}
    cfg = EngineConfig()

    res = BacktestResult(
        equity=equity_df,
        trades=trades_df,
        orders=orders_df,
        metrics=metrics,
        config=cfg,
        strategy_name="mean_reversion",
    )
    assert res.final_equity == 1000.0
    assert res.strategy_name == "mean_reversion"
    as_dict = res.to_dict()
    assert "equity" in as_dict
    assert "metrics" in as_dict

