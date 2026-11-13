import pandas as pd
import pytest

from quant_backtester.config import RiskConfig
from quant_backtester.execution.fee import FixedCommissionModel
from quant_backtester.execution.order_manager import OrderManager
from quant_backtester.execution.standard import StandardExecutionModel
from quant_backtester.models.order import OrderOrigin, OrderSide, OrderStatus, OrderType
from quant_backtester.models.quote import QuoteBar
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.portfolio import Portfolio
from quant_backtester.risk.simple import SimpleRiskManager


def test_two_phase_fill_propose_finalize_commit_and_rejection():
    account = Account(initial_balance=1000.0)
    portfolio = Portfolio(account=account)
    om = OrderManager()
    rm = SimpleRiskManager(config=RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02))
    exec_model = StandardExecutionModel(fee_model=FixedCommissionModel(amount_per_lot=7.0, mode="ROUND_TURN"))

    ts0 = pd.Timestamp("2026-01-01 00:00")
    ts1 = pd.Timestamp("2026-01-01 00:15")
    # Bar 1 opens with gap up to 2020.0
    qb1 = QuoteBar("XAUUSD", ts1, 2020.0, 2025.0, 2015.0, 2022.0, 2020.0, 2025.0, 2015.0, 2022.0)

    # Case 1: Order with fixed stop_loss_price = 2025.0 on a BUY order where fill price is 2020.0
    # (SL 2025.0 >= Fill Price 2020.0 is invalid for BUY!)
    bad_order = om.create_order(
        time=ts0,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        units=1.0,
        bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
        stop_loss_price=2025.0,
    )

    # Step 1: Propose -> does NOT touch Account, Portfolio, or Order status
    proposal = exec_model.propose_entry_open(bad_order, qb1, bar_index=1)
    assert proposal is not None
    assert account.balance == 1000.0
    assert portfolio.get_position("XAUUSD").is_flat
    assert bad_order.status == OrderStatus.PENDING

    # Step 2: Validate/Finalize against actual proposed fill_price (2020.0) -> fails!
    ok, reason, _, _ = rm.finalize_entry_sl_tp(bad_order, proposal.fill_price)
    assert ok is False
    assert reason == "INVALID_SL_AT_FILL_PRICE"

    om.reject_pending_order(bad_order, timestamp=ts1, bar_index=1, reason=reason)
    assert bad_order.status == OrderStatus.REJECTED
    assert account.balance == 1000.0
    assert len(account.ledger.entries) == 0

    # Case 2: Valid order with stop_loss_pct=0.01, take_profit_pct=0.02 -> commits cleanly once
    good_order = om.create_order(
        time=ts0,
        symbol="XAUUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01,
        units=1.0,
        bar_index=0,
        eligible_from_bar=1,
        origin=OrderOrigin.STRATEGY_ENTRY,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )
    prop2 = exec_model.propose_entry_open(good_order, qb1, bar_index=1)
    assert prop2 is not None
    ok2, _, sl_p, tp_p = rm.finalize_entry_sl_tp(good_order, prop2.fill_price)
    assert ok2 is True
    assert sl_p == pytest.approx(2020.0 * 0.99)
    assert tp_p == pytest.approx(2020.0 * 1.02)

    fill = exec_model.commit_proposal(
        prop2,
        order_manager=om,
        portfolio=portfolio,
        stop_loss_price=sl_p,
        take_profit_price=tp_p,
    )
    assert fill.price == pytest.approx(2020.0)
    assert good_order.status == OrderStatus.FILLED
    pos = portfolio.get_position("XAUUSD")
    assert pos.is_long
    assert pos.stop_loss_price == pytest.approx(2020.0 * 0.99)
    assert pos.take_profit_price == pytest.approx(2020.0 * 1.02)
    assert account.balance == pytest.approx(1000.0 - 0.035)

