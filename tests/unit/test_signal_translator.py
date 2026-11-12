import pandas as pd
import pytest

from quant_backtester.config import ReversePolicy
from quant_backtester.models.order import OrderSide
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.portfolio.position import PositionSide, PositionView
from quant_backtester.strategies.translator import SignalTranslator


def _pos(side: PositionSide, units: float = 1.0, entry_price: float = 2000.0) -> PositionView:
    if side == PositionSide.FLAT:
        return PositionView(symbol="XAUUSD", side=PositionSide.FLAT, units=0.0, entry_price=0.0, entry_time=None)
    return PositionView(
        symbol="XAUUSD",
        side=side,
        units=units,
        entry_price=entry_price,
        entry_time=pd.Timestamp("2026-01-01 00:00"),
    )


def _sig(sig_type: SignalType, directives: dict | None = None) -> Signal:
    meta = {"directives": directives} if directives else {}
    return Signal(
        time=pd.Timestamp("2026-01-01 00:15"),
        symbol="XAUUSD",
        signal_type=sig_type,
        strength=1.0,
        metadata=meta,
    )


def test_decision_table_when_flat():
    translator = SignalTranslator(default_quantity=0.01)
    flat = _pos(PositionSide.FLAT)

    # 1. FLAT + LONG -> BUY
    intent_long = translator.translate(_sig(SignalType.LONG), flat)
    assert intent_long is not None
    assert intent_long.side == OrderSide.BUY
    assert intent_long.reason == "ENTRY_LONG"

    # 2. FLAT + SHORT -> SELL
    intent_short = translator.translate(_sig(SignalType.SHORT), flat)
    assert intent_short is not None
    assert intent_short.side == OrderSide.SELL
    assert intent_short.reason == "ENTRY_SHORT"

    # 3. FLAT + EXIT_LONG / EXIT_SHORT / EXIT_ALL / FLAT -> None
    for st in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.EXIT_ALL, SignalType.FLAT):
        assert translator.translate(_sig(st), flat) is None


def test_decision_table_when_long():
    translator = SignalTranslator(default_quantity=0.01, reverse_policy=ReversePolicy.IGNORE)
    long_pos = _pos(PositionSide.LONG)

    # LONG + LONG -> None (already long)
    assert translator.translate(_sig(SignalType.LONG), long_pos) is None

    # LONG + EXIT_LONG -> close BUY (i.e. SELL)
    intent_exit = translator.translate(_sig(SignalType.EXIT_LONG), long_pos)
    assert intent_exit is not None
    assert intent_exit.side == OrderSide.SELL
    assert intent_exit.reason == "STRATEGY_EXIT"

    # LONG + EXIT_SHORT -> None
    assert translator.translate(_sig(SignalType.EXIT_SHORT), long_pos) is None

    # LONG + EXIT_ALL -> close BUY (i.e. SELL)
    intent_exit_all = translator.translate(_sig(SignalType.EXIT_ALL), long_pos)
    assert intent_exit_all is not None
    assert intent_exit_all.side == OrderSide.SELL
    assert intent_exit_all.reason == "STRATEGY_EXIT"

    # LONG + SHORT with IGNORE -> None
    assert translator.translate(_sig(SignalType.SHORT), long_pos) is None
    assert translator.signal_log[-1]["decision"] == "IGNORED"
    assert translator.signal_log[-1]["reason"] == "REVERSE_POLICY_IGNORE"


def test_decision_table_when_short():
    translator = SignalTranslator(default_quantity=0.01, reverse_policy=ReversePolicy.IGNORE)
    short_pos = _pos(PositionSide.SHORT)

    # SHORT + SHORT -> None
    assert translator.translate(_sig(SignalType.SHORT), short_pos) is None

    # SHORT + EXIT_SHORT -> close SELL (i.e. BUY)
    intent_exit = translator.translate(_sig(SignalType.EXIT_SHORT), short_pos)
    assert intent_exit is not None
    assert intent_exit.side == OrderSide.BUY
    assert intent_exit.reason == "STRATEGY_EXIT"

    # SHORT + EXIT_LONG -> None
    assert translator.translate(_sig(SignalType.EXIT_LONG), short_pos) is None

    # SHORT + EXIT_ALL -> close SELL (i.e. BUY)
    intent_exit_all = translator.translate(_sig(SignalType.EXIT_ALL), short_pos)
    assert intent_exit_all is not None
    assert intent_exit_all.side == OrderSide.BUY
    assert intent_exit_all.reason == "STRATEGY_EXIT"

    # SHORT + LONG with IGNORE -> None
    assert translator.translate(_sig(SignalType.LONG), short_pos) is None
    assert translator.signal_log[-1]["decision"] == "IGNORED"
    assert translator.signal_log[-1]["reason"] == "REVERSE_POLICY_IGNORE"


def test_reverse_policy_reverse_closes_existing_position_first():
    translator = SignalTranslator(default_quantity=0.01, reverse_policy=ReversePolicy.REVERSE)
    long_pos = _pos(PositionSide.LONG)

    intent = translator.translate(_sig(SignalType.SHORT), long_pos)
    assert intent is not None
    assert intent.side == OrderSide.SELL
    assert intent.metadata.get("reverse_to_side") == OrderSide.SELL
    assert translator.signal_log[-1]["decision"] == "REVERSE_CLOSE"


def test_signal_log_contains_required_fields():
    translator = SignalTranslator(default_quantity=0.01)
    flat = _pos(PositionSide.FLAT)
    translator.translate(_sig(SignalType.LONG), flat)
    translator.translate(_sig(SignalType.EXIT_LONG), flat)

    assert len(translator.signal_log) == 2
    required_keys = {
        "timestamp",
        "symbol",
        "signal_type",
        "position_before",
        "decision",
        "reason",
        "intent_created",
    }
    for entry in translator.signal_log:
        assert required_keys.issubset(entry.keys())
    assert translator.signal_log[0]["intent_created"] is True
    assert translator.signal_log[1]["intent_created"] is False

