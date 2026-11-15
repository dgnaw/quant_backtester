import pandas as pd
import pytest

from quant_backtester.config import InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.portfolio import Portfolio


def _make_portfolio(mode: PositionMode = PositionMode.NETTING_SCALE_IN) -> Portfolio:
    acc = Account(initial_balance=1000.0, base_currency="USD")
    inst = InstrumentConfig(symbol="XAUUSD", lot_size=0.01, contract_size=100.0)
    return Portfolio(account=acc, instrument_config=inst, default_position_mode=mode)


def test_1_trade_id_sequential_and_unique():
    pf = _make_portfolio(PositionMode.NETTING_SINGLE)
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.01, 1.0, fee=0.10),
        bar_index=1,
    )
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.SELL, 2010.0, 0.01, 1.0, fee=0.10),
        bar_index=2,
    )
    pf.on_fill(
        Fill("F-3", "O-3", pd.Timestamp("2026-01-01 00:30"), "XAUUSD", OrderSide.BUY, 2005.0, 0.01, 1.0, fee=0.10),
        bar_index=3,
    )
    pf.on_fill(
        Fill("F-4", "O-4", pd.Timestamp("2026-01-01 00:45"), "XAUUSD", OrderSide.SELL, 2015.0, 0.01, 1.0, fee=0.10),
        bar_index=4,
    )
    assert len(pf.closed_trades) == 2
    assert pf.closed_trades[0]["trade_id"] == "TRD-00001"
    assert pf.closed_trades[1]["trade_id"] == "TRD-00002"


def test_2_single_entry_links_order_and_fill_ids():
    pf = _make_portfolio(PositionMode.NETTING_SINGLE)
    pf.on_fill(
        Fill("FILL-001", "ORD-001", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.02, 2.0, fee=0.20),
        bar_index=1,
    )
    pf.on_fill(
        Fill("FILL-002", "ORD-002", pd.Timestamp("2026-01-01 00:30"), "XAUUSD", OrderSide.SELL, 2015.0, 0.02, 2.0, fee=0.20),
        bar_index=3,
    )
    tr = pf.closed_trades[0]
    assert tr["entry_order_id"] == "ORD-001"
    assert tr["entry_fill_id"] == "FILL-001"
    assert tr["entry_order_ids"] == ["ORD-001"]
    assert tr["entry_fill_ids"] == ["FILL-001"]
    assert tr["exit_order_id"] == "ORD-002"
    assert tr["exit_fill_id"] == "FILL-002"


def test_3_trade_after_scale_in_has_multiple_entry_fills_and_none_single_id():
    pf = _make_portfolio(PositionMode.NETTING_SCALE_IN)
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY, 2000.0, 0.02, 2.0, fee=0.20),
        bar_index=1,
    )
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.BUY, 2030.0, 0.03, 3.0, fee=0.30),
        bar_index=2,
    )
    pf.on_fill(
        Fill("F-3", "O-3", pd.Timestamp("2026-01-01 00:45"), "XAUUSD", OrderSide.SELL, 2040.0, 0.05, 5.0, fee=0.50),
        bar_index=4,
    )
    tr = pf.closed_trades[0]
    assert tr["entry_order_ids"] == ["O-1", "O-2"]
    assert tr["entry_fill_ids"] == ["F-1", "F-2"]
    assert tr["entry_order_id"] is None
    assert tr["entry_fill_id"] is None


def test_4_5_6_trade_allocation_matches_units_quantity_fee_and_slippage():
    pf = _make_portfolio(PositionMode.NETTING_SCALE_IN)
    pf.on_fill(
        Fill(
            "F-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.BUY,
            2000.0, 0.02, 2.0, fee=0.20, metadata={"slippage_cost": 0.40},
        ),
        bar_index=1,
    )
    pf.on_fill(
        Fill(
            "F-2", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.BUY,
            2030.0, 0.03, 3.0, fee=0.30, metadata={"slippage_cost": 0.60},
        ),
        bar_index=2,
    )
    # Partial close 2.0 units (40% of 5.0 units)
    pf.on_fill(
        Fill(
            "F-3", "O-3", pd.Timestamp("2026-01-01 00:30"), "XAUUSD", OrderSide.SELL,
            2040.0, 0.02, 2.0, fee=0.20, metadata={"slippage_cost": 0.10},
        ),
        bar_index=3,
    )
    # Full close remaining 3.0 units
    pf.on_fill(
        Fill(
            "F-4", "O-4", pd.Timestamp("2026-01-01 00:45"), "XAUUSD", OrderSide.SELL,
            2050.0, 0.03, 3.0, fee=0.30, metadata={"slippage_cost": 0.15},
        ),
        bar_index=4,
    )

    alloc_df = pf.trade_allocations_to_dataframe()
    assert len(alloc_df) == 4  # 2 allocations for TRD-00001 + 2 allocations for TRD-00002
    assert alloc_df["allocated_units"].sum() == pytest.approx(5.0)
    assert alloc_df["allocated_quantity"].sum() == pytest.approx(0.05)
    assert alloc_df["allocated_entry_fee"].sum() == pytest.approx(0.50)
    assert alloc_df["allocated_entry_slippage_cost"].sum() == pytest.approx(1.00)


def test_7_8_holding_bars_and_weighted_holding_bars():
    pf = _make_portfolio(PositionMode.NETTING_SCALE_IN)
    # Single entry at bar 2, exit at bar 7 -> holding_bars = 5, weighted_holding_bars = 5.0
    pf.on_fill(
        Fill("F-1", "O-1", pd.Timestamp("2026-01-01 00:30"), "XAUUSD", OrderSide.BUY, 2000.0, 0.01, 1.0),
        bar_index=2,
    )
    pf.on_fill(
        Fill("F-2", "O-2", pd.Timestamp("2026-01-01 01:45"), "XAUUSD", OrderSide.SELL, 2010.0, 0.01, 1.0),
        bar_index=7,
    )
    assert pf.closed_trades[0]["holding_bars"] == 5
    assert pf.closed_trades[0]["weighted_holding_bars"] == pytest.approx(5.0)

    # Scale-in: 2 units at bar 10, 3 units at bar 12 -> exit 5 units at bar 15
    # weighted_holding_bars = (2 * (15 - 10) + 3 * (15 - 12)) / 5 = (10 + 9) / 5 = 3.8
    pf.on_fill(
        Fill("F-3", "O-3", pd.Timestamp("2026-01-01 02:30"), "XAUUSD", OrderSide.BUY, 2000.0, 0.02, 2.0),
        bar_index=10,
    )
    pf.on_fill(
        Fill("F-4", "O-4", pd.Timestamp("2026-01-01 03:00"), "XAUUSD", OrderSide.BUY, 2010.0, 0.03, 3.0),
        bar_index=12,
    )
    pf.on_fill(
        Fill("F-5", "O-5", pd.Timestamp("2026-01-01 03:45"), "XAUUSD", OrderSide.SELL, 2020.0, 0.05, 5.0),
        bar_index=15,
    )
    assert pf.closed_trades[1]["weighted_holding_bars"] == pytest.approx(3.8)
    assert pf.closed_trades[1]["holding_bars"] == pytest.approx(3.8)


def test_9_10_11_12_pnl_slippage_return_pct_and_backward_compat():
    pf = _make_portfolio(PositionMode.NETTING_SINGLE)
    pf.on_fill(
        Fill(
            "F-1", "O-1", pd.Timestamp("2026-01-01 00:00"), "XAUUSD", OrderSide.SELL,
            2000.0, 0.01, 1.0, fee=0.50, metadata={"slippage_cost": 0.25},
        ),
        bar_index=1,
    )
    pf.on_fill(
        Fill(
            "F-2", "O-2", pd.Timestamp("2026-01-01 00:15"), "XAUUSD", OrderSide.BUY,
            1980.0, 0.01, 1.0, fee=0.50, metadata={"slippage_cost": 0.25},
        ),
        bar_index=2,
    )
    tr = pf.closed_trades[0]
    # SHORT 1 unit from 2000 to 1980 -> gross_pnl = +20.0, total_fee = 1.0, net_pnl = +19.0 (slippage NOT subtracted twice!)
    assert tr["gross_pnl"] == pytest.approx(20.0)
    assert tr["total_fee"] == pytest.approx(1.0)
    assert tr["net_pnl"] == pytest.approx(19.0)
    assert tr["entry_slippage_cost"] == pytest.approx(0.25)
    assert tr["exit_slippage_cost"] == pytest.approx(0.25)
    # return_pct = (19.0 / (2000.0 * 1.0)) * 100 = 0.95%
    assert tr["return_pct"] == pytest.approx(0.95)

    legacy_cols = {
        "symbol", "direction", "units", "entry_time", "exit_time",
        "entry_price", "exit_price", "entry_fee", "exit_fee",
        "gross_pnl", "net_pnl", "reason",
    }
    assert legacy_cols.issubset(set(tr.keys()))


def test_13_empty_trade_and_allocation_dataframes_have_full_schema():
    pf = _make_portfolio()
    trades_df = pf.trades_to_dataframe()
    alloc_df = pf.trade_allocations_to_dataframe()
    assert trades_df.empty
    assert alloc_df.empty
    assert "trade_id" in trades_df.columns
    assert "net_pnl" in trades_df.columns
    assert "weighted_holding_bars" in trades_df.columns
    assert "allocated_entry_fee" in alloc_df.columns
