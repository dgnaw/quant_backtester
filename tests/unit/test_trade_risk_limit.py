"""
Unit tests for Per-Trade Risk Cap (Group F - Task F2 / DF3).
Covers all 8 required test cases.
"""

from __future__ import annotations

import math
import pytest

from quant_backtester.config import (
    CommissionMode,
    InstrumentConfig,
    RiskConfig,
    SizingBasis,
    SizingMode,
)
from quant_backtester.execution.fee import FixedCommissionModel
from quant_backtester.execution.slippage import FixedSlippageModel
from quant_backtester.models.order import Order, OrderIntent, OrderSide, OrderType
from quant_backtester.risk import RiskContext, SimpleRiskManager
import pandas as pd


def _inst() -> InstrumentConfig:
    return InstrumentConfig(
        symbol="XAUUSD",
        base_currency="XAU",
        quote_currency="USD",
        contract_size=100.0,
        lot_size=0.10,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
    )


def test_fixed_lot_passes_when_within_risk_cap_pct():
    """1. Fixed lot order with risk <= max_risk_per_trade_pct is approved."""
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_pct=0.02,  # 2% of 10,000 = $200
        ),
        instrument_config=_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.15,  # 0.15 * 100 * (2000 - 1990) = $150 <= $200
        stop_loss_price=1990.0,
    )
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert dec.is_approved
    assert math.isclose(dec.estimated_risk_amount, 150.0)


def test_fixed_lot_rejected_when_exceeding_risk_cap_pct():
    """2. Fixed lot order with risk > max_risk_per_trade_pct is rejected with MAX_TRADE_RISK_EXCEEDED."""
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_pct=0.01,  # 1% of 10,000 = $100
        ),
        instrument_config=_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.15,  # $150 > $100
        stop_loss_price=1990.0,
    )
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "MAX_TRADE_RISK_EXCEEDED"


def test_fixed_lot_rejected_when_exceeding_risk_cap_amount():
    """3. Fixed lot order with risk > max_risk_per_trade_amount is rejected with MAX_TRADE_RISK_EXCEEDED."""
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_amount=120.0,
        ),
        instrument_config=_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.15,  # $150 > $120
        stop_loss_price=2010.0,
    )
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "MAX_TRADE_RISK_EXCEEDED"


def test_both_pct_and_amount_caps_uses_stricter_minimum():
    """4. When both max_risk_per_trade_pct and max_risk_per_trade_amount are set, uses min(caps)."""
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_pct=0.02,     # 2% of 10,000 = $200
            max_risk_per_trade_amount=120.0, # $120 is stricter
        ),
        instrument_config=_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.15,  # $150 > min(200, 120) = $120
        stop_loss_price=1990.0,
    )
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "MAX_TRADE_RISK_EXCEEDED"


def test_risk_cap_requires_stop_loss():
    """5. When risk cap is enabled and order has no SL, rejects with STOP_LOSS_REQUIRED_BY_RISK_CAP."""
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_amount=200.0,
            stop_loss_pct=0.0,
        ),
        instrument_config=_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.10,
    )
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "STOP_LOSS_REQUIRED_BY_RISK_CAP"


def test_risk_cap_includes_fees_and_slippage():
    """6. Order whose price risk is within cap ($100) but exceeds cap once fee + slippage are added is rejected."""
    fee_model = FixedCommissionModel(amount_per_lot=10.0, mode=CommissionMode.PER_SIDE)
    slip_model = FixedSlippageModel(0.20)
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_amount=100.0,
            include_fees_in_risk=True,
            include_slippage_in_risk=True,
        ),
        instrument_config=_inst(),
        fee_model=fee_model,
        slippage_model=slip_model,
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.10,  # Pure price loss = $100, but with slippage + fee > $100
        stop_loss_price=1990.0,
    )
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "MAX_TRADE_RISK_EXCEEDED"


def test_risk_based_sizing_capped_by_stricter_max_risk_per_trade_amount():
    """7. RISK_BASED sizing with 2% budget ($200) is rejected if max_risk_per_trade_amount is $100."""
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_pct=0.02,          # sizes to $200 (0.20 lot)
            max_risk_per_trade_amount=100.0,  # hard cap at $100
        ),
        instrument_config=_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1990.0,
    )
    dec = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec.is_approved
    assert dec.reject_reason == "MAX_TRADE_RISK_EXCEEDED"


def test_finalize_entry_sl_tp_rechecks_risk_cap_at_gapped_fill_price():
    """
    8. At signal time (ref=2000, SL=1990), 0.10 lot has $100 risk <= $120 cap.
    At next bar open, price gaps up to 2005 with fixed SL=1990 -> risk becomes $150 > $120 cap!
    finalize_entry_sl_tp rejects with MAX_TRADE_RISK_EXCEEDED.
    """
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.FIXED_LOT,
            max_risk_per_trade_amount=120.0,
        ),
        instrument_config=_inst(),
    )
    order = Order(
        order_id="ORD-00001",
        time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.10,
        units=10.0,
        stop_loss_price=1990.0,
    )
    ctx = RiskContext(current_balance=10000.0, current_equity=10000.0)
    ok, reason, _, _ = rm.finalize_entry_sl_tp(order, proposed_fill_price=2005.0, context=ctx)
    assert not ok
    assert reason == "MAX_TRADE_RISK_EXCEEDED"
