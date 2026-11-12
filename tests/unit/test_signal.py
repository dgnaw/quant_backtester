import pandas as pd
import pytest

from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.adapters import frame_to_signals


def test_signal_validation_and_immutability():
    meta_in = {"directives": {"stop_loss_pct": 0.01, "quantity": 0.05}}
    sig = Signal(
        time=pd.Timestamp("2026-01-01 10:00:00"),
        symbol="XAUUSD",
        signal_type=SignalType.LONG,
        strength=0.8,
        metadata=meta_in,
    )
    assert sig.symbol == "XAUUSD"
    assert sig.signal_type == SignalType.LONG
    assert sig.strength == pytest.approx(0.8)
    assert sig.is_entry is True
    assert sig.is_exit is False
    assert sig.metadata["directives"]["stop_loss_pct"] == 0.01

    # Mutating original dict must not mutate Signal.metadata
    meta_in["directives"]["stop_loss_pct"] = 0.99
    assert sig.metadata["directives"]["stop_loss_pct"] == 0.01

    # Signal attributes and metadata must be immutable
    with pytest.raises(Exception):
        sig.symbol = "EURUSD"
    with pytest.raises((TypeError, AttributeError)):
        sig.metadata["new_key"] = 123


def test_signal_validation_rejects_invalid_fields():
    ts = pd.Timestamp("2026-01-01 10:00:00")

    # Empty symbol
    with pytest.raises(ValueError, match="symbol"):
        Signal(time=ts, symbol="", signal_type=SignalType.LONG)
    with pytest.raises(ValueError, match="symbol"):
        Signal(time=ts, symbol="   ", signal_type=SignalType.LONG)

    # Invalid time
    with pytest.raises(ValueError, match="time"):
        Signal(time=pd.NaT, signal_type=SignalType.LONG, symbol="XAUUSD")
    with pytest.raises(ValueError, match="time"):
        Signal(time=None, signal_type=SignalType.LONG, symbol="XAUUSD")

    # Invalid signal_type
    with pytest.raises(ValueError, match="signal_type"):
        Signal(time=ts, symbol="XAUUSD", signal_type="LONG")  # type: ignore

    # Invalid strength (NaN, Inf, negative, > 1.0, or bool)
    with pytest.raises(ValueError, match="strength"):
        Signal(time=ts, symbol="XAUUSD", signal_type=SignalType.LONG, strength=float("nan"))
    with pytest.raises(ValueError, match="strength"):
        Signal(time=ts, symbol="XAUUSD", signal_type=SignalType.LONG, strength=float("inf"))
    with pytest.raises(ValueError, match="strength"):
        Signal(time=ts, symbol="XAUUSD", signal_type=SignalType.LONG, strength=-0.1)
    with pytest.raises(ValueError, match="strength"):
        Signal(time=ts, symbol="XAUUSD", signal_type=SignalType.LONG, strength=1.5)


def test_signal_semantics_and_adapter_directives():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=3, freq="15min"),
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [95.0, 96.0, 97.0],
        "close": [101.0, 102.0, 103.0],
        "entry_signal": [1, 0, 0],
        "exit_long_signal": [False, True, True],
        "exit_short_signal": [False, False, True],
        "quantity": [0.1, None, None],
        "stop_loss_price": [95.0, None, None],
    })
    signals = frame_to_signals(df, symbol="XAUUSD")
    assert signals[0].signal_type == SignalType.LONG
    assert signals[0].metadata["directives"]["quantity"] == 0.1
    assert signals[0].metadata["directives"]["stop_loss_price"] == 95.0
    assert signals[1].signal_type == SignalType.EXIT_LONG
    assert signals[2].signal_type == SignalType.EXIT_ALL

