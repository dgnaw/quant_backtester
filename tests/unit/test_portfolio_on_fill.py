import pandas as pd
import pytest

from quant_backtester.config import InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.portfolio import Portfolio


def _make_pf(mode: PositionMode = PositionMode.NETTING_SCALE_IN) -> Portfolio:
    return Portfolio(
        account=Account(initial_balance=1000.0, base_currency="USD"),
        instrument_config=InstrumentConfig(symbol="XAUUSD", lot_size=0.01, contract_size=100.0),
        default_position_mode=mode,
    )


def test_all_10_branches_of_portfolio_on_fill():
    pf = _make_pf(PositionMode.NETTING_SCALE_IN)

    # 1. FLAT + BUY -> Open LONG (2 units @ 2000)
    pf.on_fill(Fill("F-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.02, 2.0, fee=0.20))
    assert pf.get_position("XAUUSD").is_long

    # 3. LONG + BUY -> Scale-in LONG (+3 units @ 2010 -> total 5 units @ 2006)
    pf.on_fill(Fill("F-2", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.BUY, 2010.0, 0.03, 3.0, fee=0.30))
    assert pf.get_position("XAUUSD").units == pytest.approx(5.0)
    assert pf.get_position("XAUUSD").average_entry_price == pytest.approx(2006.0)

    # 5. LONG + SELL (< units) -> Partial Close (-2 units @ 2016 -> remaining 3 units)
    pf.on_fill(Fill("F-3", "O-3", pd.Timestamp("2026-01-01 00:30"), "XAUUSD", OrderSide.SELL, 2016.0, 0.02, 2.0, fee=0.20))
    assert pf.get_position("XAUUSD").units == pytest.approx(3.0)
    assert len(pf.closed_trades) == 1

    # 6. LONG + SELL (== units) -> Full Close (-3 units @ 2020 -> FLAT)
    pf.on_fill(Fill("F-4", "O-4", pd.Timestamp("2026-01-01 00:45"), "XAUUSD", OrderSide.SELL, 2020.0, 0.03, 3.0, fee=0.30))
    assert pf.get_position("XAUUSD").is_flat
    assert "XAUUSD" not in pf.active_positions
    assert len(pf.closed_trades) == 2

    # 2. FLAT + SELL -> Open SHORT (2 units @ 2020)
    pf.on_fill(Fill("F-5", "O-5", pd.Timestamp("2026-01-01 01:00"), "XAUUSD", OrderSide.SELL, 2020.0, 0.02, 2.0, fee=0.20))
    assert pf.get_position("XAUUSD").is_short

    # 4. SHORT + SELL -> Scale-in SHORT (+2 units @ 2010 -> total 4 units @ 2015)
    pf.on_fill(Fill("F-6", "O-6", pd.Timestamp("2026-01-01 01:15"), "XAUUSD", OrderSide.SELL, 2010.0, 0.02, 2.0, fee=0.20))
    assert pf.get_position("XAUUSD").units == pytest.approx(4.0)
    assert pf.get_position("XAUUSD").average_entry_price == pytest.approx(2015.0)

    # 8. SHORT + BUY (< units) -> Partial Close (-1 unit @ 2005 -> remaining 3 units)
    pf.on_fill(Fill("F-7", "O-7", pd.Timestamp("2026-01-01 01:30"), "XAUUSD", OrderSide.BUY, 2005.0, 0.01, 1.0, fee=0.10))
    assert pf.get_position("XAUUSD").units == pytest.approx(3.0)

    # 9. SHORT + BUY (== units) -> Full Close (-3 units @ 2000 -> FLAT)
    pf.on_fill(Fill("F-8", "O-8", pd.Timestamp("2026-01-01 01:45"), "XAUUSD", OrderSide.BUY, 2000.0, 0.03, 3.0, fee=0.30))
    assert pf.get_position("XAUUSD").is_flat

    # 7 & 10. Over-Close Reverse: Open LONG 2 units, then SELL 5 units -> Closes LONG 2, Opens SHORT 3; then BUY 6 units -> Closes SHORT 3, Opens LONG 3
    pf.on_fill(Fill("F-9", "O-9", pd.Timestamp("2026-01-01 02:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.02, 2.0, fee=0.20))
    pf.on_fill(
        Fill(
            "F-10", "O-10", pd.Timestamp("2026-01-01 02:15"), "XAUUSD", OrderSide.SELL,
            2010.0, 0.05, 5.0, fee=0.50, metadata={"slippage_cost": 1.00},
        )
    )
    pos_short = pf.get_position("XAUUSD")
    assert pos_short.is_short
    assert pos_short.units == pytest.approx(3.0)
    assert pos_short.entry_fee == pytest.approx(0.30)  # 3/5 of 0.50
    assert pos_short.entry_slippage_cost == pytest.approx(0.60)  # 3/5 of 1.00

    pf.on_fill(
        Fill(
            "F-11", "O-11", pd.Timestamp("2026-01-01 02:30"), "XAUUSD", OrderSide.BUY,
            1995.0, 0.06, 6.0, fee=0.60, metadata={"slippage_cost": 1.20},
        )
    )
    pos_long = pf.get_position("XAUUSD")
    assert pos_long.is_long
    assert pos_long.units == pytest.approx(3.0)
    assert pos_long.entry_fee == pytest.approx(0.30)  # 3/6 of 0.60
    assert pos_long.entry_slippage_cost == pytest.approx(0.60)  # 3/6 of 1.20


def test_duplicate_fill_idempotency_and_atomic_rollback():
    pf = _make_pf(PositionMode.NETTING_SINGLE)
    f1 = Fill("F-DUP-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.01, 1.0, fee=0.25)
    pf.on_fill(f1)

    bal_before = pf.account.balance
    ledger_len_before = len(pf.account.ledger.entries)
    pos_units_before = pf.get_position("XAUUSD").units

    # Duplicate fill_id must be rejected without changing any state
    with pytest.raises(ValueError, match="DUPLICATE_FILL"):
        pf.on_fill(f1)

    assert pf.account.balance == pytest.approx(bal_before)
    assert len(pf.account.ledger.entries) == ledger_len_before
    assert pf.get_position("XAUUSD").units == pytest.approx(pos_units_before)

    # Invalid scale-in fill in NETTING_SINGLE mode must not mark fill_id as processed and must leave state untouched
    f_bad = Fill("F-BAD-1", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.BUY, 2005.0, 0.01, 1.0, fee=0.25)
    with pytest.raises(ValueError, match="SCALE_IN_NOT_ALLOWED"):
        pf.on_fill(f_bad)

    assert "F-BAD-1" not in pf._processed_fill_ids
    assert pf.account.balance == pytest.approx(bal_before)
    assert len(pf.account.ledger.entries) == ledger_len_before
    assert pf.get_position("XAUUSD").units == pytest.approx(pos_units_before)
