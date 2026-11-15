import pandas as pd
import pytest

from quant_backtester.config import EngineConfig, InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.portfolio import Portfolio
from quant_backtester.portfolio.position import PositionSide, PositionView
from quant_backtester.strategies.translator import SignalTranslator


def test_netting_single_rejects_scale_in_and_netting_scale_in_allows():
    inst = InstrumentConfig(symbol="XAUUSD", lot_size=0.01, contract_size=100.0)
    pf_single = Portfolio(
        account=Account(1000.0),
        instrument_config=inst,
        default_position_mode=PositionMode.NETTING_SINGLE,
    )
    pf_single.on_fill(Fill("F-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.01, 1.0))
    with pytest.raises(ValueError, match="SCALE_IN_NOT_ALLOWED"):
        pf_single.on_fill(Fill("F-2", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.BUY, 2005.0, 0.01, 1.0))

    pf_scale = Portfolio(
        account=Account(1000.0),
        instrument_config=inst,
        default_position_mode=PositionMode.NETTING_SCALE_IN,
    )
    pf_scale.on_fill(Fill("F-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.01, 1.0))
    pf_scale.on_fill(Fill("F-2", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.BUY, 2010.0, 0.02, 2.0))
    pos = pf_scale.get_position("XAUUSD")
    assert pos.quantity == pytest.approx(0.03)
    assert pos.units == pytest.approx(3.0)


def test_default_position_mode_and_symbol_override():
    cfg = EngineConfig(
        default_position_mode=PositionMode.NETTING_SINGLE,
        position_mode_by_symbol={"EURUSD": PositionMode.NETTING_SCALE_IN},
    )
    assert cfg.get_position_mode("XAUUSD") == PositionMode.NETTING_SINGLE
    assert cfg.get_position_mode("EURUSD") == PositionMode.NETTING_SCALE_IN

    pf = Portfolio(
        account=Account(1000.0),
        default_position_mode=cfg.default_position_mode,
        position_mode_by_symbol=cfg.position_mode_by_symbol,
    )
    assert pf.get_position_mode("XAUUSD") == PositionMode.NETTING_SINGLE
    assert pf.get_position_mode("EURUSD") == PositionMode.NETTING_SCALE_IN


def test_translator_respects_position_mode():
    pos_long = PositionView(symbol="XAUUSD", side=PositionSide.LONG, quantity=0.01, units=1.0, entry_price=2000.0)
    sig_long = Signal(
        time=pd.Timestamp("2026-01-01 00:15"),
        symbol="XAUUSD",
        signal_type=SignalType.LONG,
        metadata={"directives": {"quantity": 0.02}},
    )

    tr_single = SignalTranslator(default_quantity=0.01, position_mode=PositionMode.NETTING_SINGLE)
    assert tr_single.translate(sig_long, pos_long) is None
    assert tr_single.signal_log[-1]["reason"] == "ALREADY_LONG"

    tr_scale = SignalTranslator(default_quantity=0.01, position_mode=PositionMode.NETTING_SCALE_IN)
    intent = tr_scale.translate(sig_long, pos_long)
    assert intent is not None
    assert intent.side == OrderSide.BUY
    assert intent.quantity == pytest.approx(0.02)
