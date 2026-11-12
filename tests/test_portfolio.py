import pytest
import pandas as pd
from quant_backtester.portfolio.position import Position, PositionSide
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.ledger import Ledger, LedgerEntry, TransactionType
from quant_backtester.portfolio.portfolio import Portfolio
from quant_backtester.models.order import OrderSide
from quant_backtester.models.fill import Fill


def test_position_long_pnl():
    pos = Position(symbol="XAUUSD")
    assert pos.is_flat
    assert pos.units == 0.0

    # Mua 1.0 unit tại giá 2000.0, phí 0.30
    pos.open(side=PositionSide.LONG, price=2000.0, units=1.0, timestamp=pd.Timestamp("2026-01-01 00:00"), fee=0.30)
    assert pos.is_long
    assert pos.entry_price == 2000.0
    assert pos.units == 1.0
    assert pos.entry_fee == 0.30

    # Giá thị trường tăng lên 2010.0 -> Unrealized PnL = (2010 - 2000) * 1.0 = +10.0
    assert pos.unrealized_pnl(2010.0) == pytest.approx(10.0)

    # Đóng vị thế tại giá 2015.0, exit fee 0.30
    trade = pos.close(exit_price=2015.0, timestamp=pd.Timestamp("2026-01-01 01:00"), fee=0.30, reason="TAKE_PROFIT")
    assert pos.is_flat
    assert trade["gross_pnl"] == pytest.approx(15.0)
    assert trade["entry_fee"] == pytest.approx(0.30)
    assert trade["exit_fee"] == pytest.approx(0.30)
    assert trade["net_pnl"] == pytest.approx(14.40)
    assert trade["reason"] == "TAKE_PROFIT"


def test_position_short_pnl():
    pos = Position(symbol="XAUUSD")
    # Bán khống 1.0 unit tại giá 2000.0
    pos.open(side=PositionSide.SHORT, price=2000.0, units=1.0, timestamp=pd.Timestamp("2026-01-01 00:00"), fee=0.30)
    assert pos.is_short

    # Giá giảm xuống 1980.0 -> Unrealized PnL = (2000 - 1980) * 1.0 = +20.0
    assert pos.unrealized_pnl(1980.0) == pytest.approx(20.0)

    # Giá tăng lên 2005.0 -> Unrealized PnL = (2000 - 2005) * 1.0 = -5.0
    assert pos.unrealized_pnl(2005.0) == pytest.approx(-5.0)


def test_account_and_ledger():
    ledger = Ledger()
    account = Account(initial_balance=1000.0, base_currency="USD", ledger=ledger)
    assert account.balance == 1000.0
    assert account.equity == 1000.0

    # Nạp tiền
    account.deposit(500.0, timestamp=pd.Timestamp("2026-01-01 00:00"), reason="DEPOSIT")
    assert account.balance == 1500.0

    # Trừ phí
    account.deduct_fee(2.0, timestamp=pd.Timestamp("2026-01-01 00:05"), reason="COMMISSION")
    assert account.balance == 1498.0

    # Khớp lệnh Fill
    fill_buy = Fill(
        fill_id="F1",
        order_id="O1",
        time=pd.Timestamp("2026-01-01 00:15"),
        symbol="XAUUSD",
        side=OrderSide.BUY,
        price=2000.0,
        quantity=0.01,
        units=1.0,
        fee=0.30,
    )
    portfolio = Portfolio(account=account)
    portfolio.on_fill(fill_buy)

    assert account.balance == pytest.approx(1497.70)  # 1498 - 0.30 fee
    # Giả định giá nến hiện tại là 2020.0 -> equity = balance + 20.0 = 1517.70
    equity = portfolio.get_equity(current_prices={"XAUUSD": 2020.0})
    assert equity == pytest.approx(1517.70)

