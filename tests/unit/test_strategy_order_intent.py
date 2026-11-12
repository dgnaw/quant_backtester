import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig, InstrumentConfig, RiskConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.order import OrderIntent, OrderSide
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.portfolio.position import PositionSide, PositionView
from quant_backtester.risk.simple import SimpleRiskManager
from quant_backtester.strategies.base import Strategy, StrategyMode
from quant_backtester.strategies.context import StrategyContext
from quant_backtester.strategies.translator import SignalTranslator


def test_order_intent_default_quantity_is_none():
    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY)
    assert intent.quantity is None
    assert intent.stop_loss_price is None
    assert intent.take_profit_price is None
    assert intent.stop_loss_pct is None
    assert intent.take_profit_pct is None
    assert intent.trailing_stop_pct is None


def test_translator_does_not_mutate_position_and_maps_directives():
    translator = SignalTranslator(default_quantity=None)
    pos = PositionView(symbol="XAUUSD", side=PositionSide.FLAT, units=0.0, entry_price=0.0)
    sig = Signal(
        time=pd.Timestamp("2026-01-01 00:00"),
        symbol="XAUUSD",
        signal_type=SignalType.LONG,
        metadata={"directives": {"quantity": 0.05, "stop_loss_price": 1980.0, "take_profit_price": 2040.0}},
    )
    intent = translator.translate(sig, pos)
    assert intent is not None
    assert intent.side == OrderSide.BUY
    assert intent.quantity == pytest.approx(0.05)
    assert intent.stop_loss_price == pytest.approx(1980.0)
    assert intent.take_profit_price == pytest.approx(2040.0)
    assert pos.is_flat


def test_risk_manager_directive_rules_validation():
    inst_cfg = InstrumentConfig(symbol="XAUUSD", lot_size=0.02, contract_size=100.0, min_lot=0.01, max_lot=5.0, lot_step=0.01)
    risk_cfg = RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02)
    rm = SimpleRiskManager(config=risk_cfg, instrument_config=inst_cfg)

    # 1. Fallback quantity and SL/TP from RiskConfig & InstrumentConfig when intent has None
    intent_empty = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, quantity=None)
    approved = rm.approve_order_intent(intent_empty, 1000.0, 1000.0, reference_price=2000.0)
    assert approved is not None
    assert approved.quantity == pytest.approx(0.02)
    assert approved.stop_loss_pct == pytest.approx(0.01)
    assert approved.take_profit_pct == pytest.approx(0.02)

    # 2. Explicit quantity and SL/TP by price (valid for BUY: SL < 2000 < TP)
    intent_price = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        quantity=0.10,
        stop_loss_price=1970.0,
        take_profit_price=2050.0,
    )
    approved_p = rm.approve_order_intent(intent_price, 1000.0, 1000.0, reference_price=2000.0)
    assert approved_p is not None
    assert approved_p.quantity == pytest.approx(0.10)
    assert approved_p.stop_loss_price == pytest.approx(1970.0)
    assert approved_p.take_profit_price == pytest.approx(2050.0)

    # 3. Conflict between stop_loss_price and stop_loss_pct -> Rejected
    intent_conflict_sl = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        stop_loss_price=1980.0,
        stop_loss_pct=0.01,
    )
    assert rm.approve_order_intent(intent_conflict_sl, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "CONFLICTING_SL_DIRECTIVES"

    # 4. Conflict between take_profit_price and take_profit_pct -> Rejected
    intent_conflict_tp = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        take_profit_price=2040.0,
        take_profit_pct=0.02,
    )
    assert rm.approve_order_intent(intent_conflict_tp, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "CONFLICTING_TP_DIRECTIVES"

    # 5. Wrong-side SL/TP for BUY (SL >= ref_price or TP <= ref_price)
    intent_bad_buy_sl = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, stop_loss_price=2010.0)
    assert rm.approve_order_intent(intent_bad_buy_sl, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "INVALID_SL_SIDE"

    intent_bad_buy_tp = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, take_profit_price=1990.0)
    assert rm.approve_order_intent(intent_bad_buy_tp, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "INVALID_TP_SIDE"

    # 6. Wrong-side SL/TP for SELL (SL <= ref_price or TP >= ref_price)
    intent_bad_sell_sl = OrderIntent(symbol="XAUUSD", side=OrderSide.SELL, stop_loss_price=1990.0)
    assert rm.approve_order_intent(intent_bad_sell_sl, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "INVALID_SL_SIDE"

    intent_bad_sell_tp = OrderIntent(symbol="XAUUSD", side=OrderSide.SELL, take_profit_price=2010.0)
    assert rm.approve_order_intent(intent_bad_sell_tp, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "INVALID_TP_SIDE"

    # 7. Quantity violates min_lot / max_lot / lot_step
    intent_small_lot = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, quantity=0.001)
    assert rm.approve_order_intent(intent_small_lot, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "INVALID_LOT_SIZE"

    intent_large_lot = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, quantity=10.0)
    assert rm.approve_order_intent(intent_large_lot, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "INVALID_LOT_SIZE"

    intent_bad_step = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, quantity=0.015)
    assert rm.approve_order_intent(intent_bad_step, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "INVALID_LOT_STEP"

    # 8. Unsupported directive trailing_stop_pct in V1.0 -> Rejected with UNSUPPORTED_DIRECTIVE
    intent_trailing = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, trailing_stop_pct=0.01)
    assert rm.approve_order_intent(intent_trailing, 1000.0, 1000.0, reference_price=2000.0) is None
    assert rm.last_reject_reason == "UNSUPPORTED_DIRECTIVE"


class RejectedDirectiveStrategy(Strategy):
    name = "rejected_directive_strategy"
    mode = StrategyMode.EVENT_DRIVEN

    def _initial_state(self):
        return {"reject_reasons": [], "saw_position_after_reject": None}

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        if context.bar_index == 0:
            # Emit unsupported trailing_stop_pct -> RiskManager must reject at bar 1 open
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={"directives": {"trailing_stop_pct": 0.01}},
            )
        if context.bar_index == 1:
            self.state["saw_position_after_reject"] = context.position.side
        return None

    def on_intent_rejected(self, intent, reason: str) -> None:
        self.state["reject_reasons"].append(reason)


def test_rejected_intent_does_not_open_position_and_notifies_hook():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=4, freq="15min"),
        "open": [2000.0, 2005.0, 2010.0, 2015.0],
        "high": [2005.0, 2010.0, 2015.0, 2020.0],
        "low": [1995.0, 2000.0, 2005.0, 2010.0],
        "close": [2002.0, 2008.0, 2012.0, 2018.0],
    })
    strat = RejectedDirectiveStrategy()
    res = BacktestEngine(BacktestConfig()).run(df, strat)

    assert len(res.trades) == 0
    assert strat.state["reject_reasons"] == ["UNSUPPORTED_DIRECTIVE"]
    assert strat.state["saw_position_after_reject"] == PositionSide.FLAT

