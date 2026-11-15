"""
Unit tests for Position Sizing & RiskEstimator (Group F - Task F1 / DF2).
Covers all 14 required test cases.
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
from quant_backtester.execution.fee import FixedCommissionModel, PercentageFeeModel
from quant_backtester.execution.slippage import FixedSlippageModel
from quant_backtester.models.order import OrderIntent, OrderSide, OrderType
from quant_backtester.models.quote import QuoteBar
from quant_backtester.risk import RiskContext, SimpleRiskManager
import pandas as pd


def _xau_inst(
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    lot_step: float = 0.01,
    contract_size: float = 100.0,
    lot_size: float = 0.10,
) -> InstrumentConfig:
    return InstrumentConfig(
        symbol="XAUUSD",
        base_currency="XAU",
        quote_currency="USD",
        contract_size=contract_size,
        lot_size=lot_size,
        min_lot=min_lot,
        max_lot=max_lot,
        lot_step=lot_step,
    )


def test_fixed_lot_uses_intent_quantity():
    """1. Fixed lot sizing uses explicit intent.quantity when valid."""
    rm = SimpleRiskManager(
        config=RiskConfig(sizing_mode=SizingMode.FIXED_LOT),
        instrument_config=_xau_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.25,
    )
    approved = rm.approve_order_intent(intent, current_balance=10000.0, current_equity=10000.0)
    assert approved is not None
    assert math.isclose(approved.quantity, 0.25)


def test_fixed_lot_falls_back_to_instrument_lot_size():
    """2. Fixed lot sizing falls back to instrument_config.lot_size when intent.quantity is None."""
    rm = SimpleRiskManager(
        config=RiskConfig(sizing_mode=SizingMode.FIXED_LOT),
        instrument_config=_xau_inst(lot_size=0.15),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=None,
    )
    approved = rm.approve_order_intent(intent, current_balance=10000.0, current_equity=10000.0)
    assert approved is not None
    assert math.isclose(approved.quantity, 0.15)


def test_fixed_lot_rejects_invalid_lot_step_and_bounds():
    """3. Fixed lot sizing rejects invalid lot step and out-of-bounds quantities."""
    rm = SimpleRiskManager(
        config=RiskConfig(sizing_mode=SizingMode.FIXED_LOT),
        instrument_config=_xau_inst(min_lot=0.05, max_lot=5.0, lot_step=0.05),
    )
    # Invalid step (0.07 not a multiple of 0.05)
    intent_step = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.07,
    )
    assert rm.approve_order_intent(intent_step, 10000.0, 10000.0) is None
    assert rm.last_reject_reason == "INVALID_LOT_STEP"

    # Below min_lot
    intent_min = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
    )
    assert rm.approve_order_intent(intent_min, 10000.0, 10000.0) is None
    assert rm.last_reject_reason == "INVALID_LOT_SIZE"


def test_risk_based_sizing_with_stop_loss_price():
    """
    4. Risk-based sizing with explicit stop_loss_price:
    Equity = 10,000, risk_per_trade_pct = 0.01 -> budget = $100.
    Entry = 2000.0, SL = 1990.0 -> stop_distance = 10.0.
    Loss per 1.0 lot (100 oz) = $1000 -> sized_lots = 100 / 1000 = 0.10 lot.
    """
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            sizing_basis=SizingBasis.EQUITY,
            risk_per_trade_pct=0.01,
        ),
        instrument_config=_xau_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1990.0,
    )
    decision = rm.evaluate_order_intent(
        intent, current_balance=10000.0, current_equity=10000.0, reference_price=2000.0
    )
    assert decision.is_approved
    assert math.isclose(decision.sized_quantity, 0.10)
    assert math.isclose(decision.estimated_risk_amount, 100.0)


def test_risk_based_sizing_with_stop_loss_pct():
    """
    5. Risk-based sizing with stop_loss_pct:
    Equity = 10,000, risk_per_trade_amount = $200.
    Entry = 2000.0, stop_loss_pct = 0.005 (0.5% -> 10.0 pts -> SL = 1990.0).
    Loss per 1.0 lot = $1000 -> sized_lots = 200 / 1000 = 0.20 lot.
    """
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=200.0,
        ),
        instrument_config=_xau_inst(),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        stop_loss_pct=0.005,
    )
    decision = rm.evaluate_order_intent(
        intent, current_balance=10000.0, current_equity=10000.0, reference_price=2000.0
    )
    assert decision.is_approved
    assert math.isclose(decision.sized_quantity, 0.20)
    assert math.isclose(decision.estimated_risk_amount, 200.0)


def test_risk_based_sizing_equity_vs_balance_basis():
    """6. SizingBasis.EQUITY vs SizingBasis.BALANCE produces proportional lot sizes."""
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1990.0,
    )
    rm_eq = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            sizing_basis=SizingBasis.EQUITY,
            risk_per_trade_pct=0.01,
        ),
        instrument_config=_xau_inst(),
    )
    rm_bal = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            sizing_basis=SizingBasis.BALANCE,
            risk_per_trade_pct=0.01,
        ),
        instrument_config=_xau_inst(),
    )
    # Balance = 10,000, Equity = 15,000
    dec_eq = rm_eq.evaluate_order_intent(intent, 10000.0, 15000.0, reference_price=2000.0)
    dec_bal = rm_bal.evaluate_order_intent(intent, 10000.0, 15000.0, reference_price=2000.0)
    assert math.isclose(dec_eq.sized_quantity, 0.15)
    assert math.isclose(dec_bal.sized_quantity, 0.10)


def test_risk_based_sizing_floors_to_lot_step_never_rounds_up():
    """
    7. Floor to lot_step: raw_lots = 0.129 with lot_step = 0.01 -> 0.12 (never 0.13).
    """
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=129.0,
        ),
        instrument_config=_xau_inst(lot_step=0.01),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1990.0,  # $1000 loss per lot
    )
    decision = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert decision.is_approved
    assert math.isclose(decision.sized_quantity, 0.12)
    assert decision.estimated_risk_amount <= 129.0


def test_risk_based_sizing_rejects_when_below_min_lot():
    """
    8. When floored lot < min_lot, rejects with SIZED_LOT_BELOW_MIN (never rounds up to min_lot).
    """
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=5.0,  # raw_lots = 0.005 < min_lot 0.01
        ),
        instrument_config=_xau_inst(min_lot=0.01, lot_step=0.01),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1990.0,
    )
    decision = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not decision.is_approved
    assert decision.reject_reason == "SIZED_LOT_BELOW_MIN"


def test_risk_based_sizing_clamp_vs_reject_above_max_lot():
    """
    9. Above max_lot: clamp_to_max_lot=True clamps to max_lot; clamp_to_max_lot=False rejects with SIZED_LOT_ABOVE_MAX.
    """
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1999.0,  # $100 per lot; budget $500 -> 5.0 lots > max_lot 2.0
    )
    rm_clamp = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=500.0,
            clamp_to_max_lot=True,
        ),
        instrument_config=_xau_inst(max_lot=2.0),
    )
    dec_clamp = rm_clamp.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert dec_clamp.is_approved
    assert math.isclose(dec_clamp.sized_quantity, 2.0)

    rm_no_clamp = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=500.0,
            clamp_to_max_lot=False,
        ),
        instrument_config=_xau_inst(max_lot=2.0),
    )
    dec_reject = rm_no_clamp.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not dec_reject.is_approved
    assert dec_reject.reject_reason == "SIZED_LOT_ABOVE_MAX"


def test_risk_based_sizing_rejects_when_stop_loss_missing():
    """10. Risk-based sizing without SL in intent or config rejects with STOP_LOSS_REQUIRED_FOR_RISK_SIZING."""
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_pct=0.01,
            stop_loss_pct=0.0,
        ),
        instrument_config=_xau_inst(),
    )
    intent = OrderIntent(symbol="XAUUSD", side=OrderSide.BUY, order_type=OrderType.MARKET)
    decision = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert not decision.is_approved
    assert decision.reject_reason == "STOP_LOSS_REQUIRED_FOR_RISK_SIZING"


def test_risk_based_sizing_includes_fees_and_slippage_step_down_loop():
    """
    11. Step-Down Verification Loop with FixedCommissionModel(min_fee) & Slippage:
    Ensures total_risk (price stop + slippage + entry fee + exit fee) <= risk_budget.
    """
    fee_model = FixedCommissionModel(
        amount_per_lot=7.0,
        mode=CommissionMode.PER_SIDE,
        min_fee=5.0,  # Flat min_fee $5 per side ($10 round-turn) for small lots!
    )
    slip_model = FixedSlippageModel(0.50)  # $0.50 slippage on entry and $0.50 on SL exit
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=100.0,
            include_fees_in_risk=True,
            include_slippage_in_risk=True,
        ),
        instrument_config=_xau_inst(min_lot=0.01, lot_step=0.01),
        fee_model=fee_model,
        slippage_model=slip_model,
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1990.0,
    )
    decision = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=2000.0)
    assert decision.is_approved
    # With entry slippage +0.50 (2000.50) and exit slippage -0.50 (1989.50), stop_distance = 11.0 ($1100/lot)
    # Plus min_fee = $5 entry + $5 exit = $10 round-turn.
    # Remaining budget for price risk = $90 -> 90 / 1100 = 0.0818 lots -> 0.08 lots.
    # Total risk at 0.08 lots = 0.08 * 1100 + 10 = $98.0 <= $100.0!
    assert math.isclose(decision.sized_quantity, 0.08)
    assert decision.estimated_risk_amount <= 100.0 + 1e-9
    assert math.isclose(decision.estimated_risk_amount, 98.0)


def test_risk_based_sizing_with_quote_bar_spread():
    """12. RiskEstimator uses QuoteBar Bid/Ask when provided in RiskContext."""
    qb = QuoteBar(
        time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        symbol="XAUUSD",
        bid_open=1999.0,
        bid_high=2005.0,
        bid_low=1995.0,
        bid_close=2000.0,
        ask_open=2000.0,
        ask_high=2006.0,
        ask_low=1996.0,
        ask_close=2001.0,
    )
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=110.0,
        ),
        instrument_config=_xau_inst(),
    )
    # BUY market at Bar Close uses ask_close = 2001.0; SL = 1990.0 -> distance = 11.0 ($1100/lot)
    intent = OrderIntent(
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=1990.0,
    )
    ctx = RiskContext(quote_bar=qb, current_balance=10000.0, current_equity=10000.0)
    decision = rm.evaluate_order_intent(intent, 10000.0, 10000.0, context=ctx)
    assert decision.is_approved
    assert math.isclose(decision.sized_quantity, 0.10)


def test_risk_based_sizing_currency_conversion_and_missing_fx_rejection():
    """
    13. Quote currency != Account currency:
    Rejects with CURRENCY_CONVERSION_UNAVAILABLE if fx_rates missing for EURGBP on USD account,
    and succeeds when valid FX rate is provided.
    """
    eurgbp = InstrumentConfig(
        symbol="EURGBP",
        base_currency="EUR",
        quote_currency="GBP",
        contract_size=100_000.0,
        lot_size=0.10,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
    )
    rm = SimpleRiskManager(
        config=RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_amount=125.0,
        ),
        instrument_config=eurgbp,
        account_currency="USD",
    )
    intent = OrderIntent(
        symbol="EURGBP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        stop_loss_price=0.8400,
    )
    # Missing FX rate GBPUSD -> rejected
    dec_no_fx = rm.evaluate_order_intent(intent, 10000.0, 10000.0, reference_price=0.8500)
    assert not dec_no_fx.is_approved
    assert dec_no_fx.reject_reason == "CURRENCY_CONVERSION_UNAVAILABLE"

    # Provide GBPUSD = 1.25 -> 1 lot stop loss = 0.0100 * 100,000 GBP = 1000 GBP = $1250 USD
    # Budget $125 USD -> 0.10 lot
    ctx_fx = RiskContext(fx_rates={"GBPUSD": 1.25}, current_balance=10000.0, current_equity=10000.0)
    dec_fx = rm.evaluate_order_intent(
        intent, 10000.0, 10000.0, reference_price=0.8500, context=ctx_fx
    )
    assert dec_fx.is_approved
    assert math.isclose(dec_fx.sized_quantity, 0.10)


def test_conflicting_risk_budget_config_raises_error():
    """14. Setting both risk_per_trade_pct and risk_per_trade_amount raises ValueError(CONFLICTING_RISK_BUDGET_CONFIG)."""
    with pytest.raises(ValueError, match="CONFLICTING_RISK_BUDGET_CONFIG"):
        RiskConfig(
            sizing_mode=SizingMode.RISK_BASED,
            risk_per_trade_pct=0.01,
            risk_per_trade_amount=100.0,
        )
