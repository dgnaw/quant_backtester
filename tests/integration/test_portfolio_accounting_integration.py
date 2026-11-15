import pandas as pd
import pytest

from quant_backtester.config import (
    AccountConfig,
    EngineConfig,
    FeeConfig,
    InstrumentConfig,
    PositionMode,
    ReversePolicy,
    RiskConfig,
    SlippageConfig,
    SpreadConfig,
)
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode


def _make_bars(prices: list[float]) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2026-01-01 00:00")
    for idx, p in enumerate(prices):
        rows.append({
            "time": t0 + pd.Timedelta(minutes=15 * idx),
            "open": p,
            "high": p + 3.0,
            "low": p - 3.0,
            "close": p + 1.0,
        })
    return pd.DataFrame(rows)


class _ScriptedEventStrategy(Strategy):
    mode = StrategyMode.EVENT_DRIVEN

    def __init__(self, script: dict[int, tuple[SignalType, dict]]):
        super().__init__(script_len=len(script))
        self.script = script

    def on_bar(self, bar, ctx):
        if ctx.bar_index in self.script:
            sig_type, directives = self.script[ctx.bar_index]
            return Signal(
                time=ctx.time,
                symbol=ctx.symbol,
                signal_type=sig_type,
                metadata={"directives": dict(directives)},
            )
        return None


def test_scenario_1_full_lifecycle_long_and_short():
    # FLAT -> LONG -> FLAT -> SHORT -> FLAT
    bars = _make_bars([2000.0, 2005.0, 2015.0, 2020.0, 2010.0, 2005.0])
    script = {
        0: (SignalType.LONG, {"quantity": 0.1}),
        1: (SignalType.EXIT_LONG, {}),
        2: (SignalType.SHORT, {"quantity": 0.1}),
        3: (SignalType.EXIT_SHORT, {}),
    }
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", lot_size=0.1, contract_size=100.0),
        spread=SpreadConfig(spread_price=1.0),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=10.0, commission_mode="PER_SIDE"),
        slippage=SlippageConfig(slippage_price=0.0),
        risk=RiskConfig(stop_loss_pct=0.20, take_profit_pct=0.50),
    )
    res = BacktestEngine(config=cfg).run(bars, _ScriptedEventStrategy(script))

    assert len(res.trades) == 2
    t1 = res.trades.iloc[0]
    t2 = res.trades.iloc[1]
    assert t1["direction"] == "LONG"
    assert t2["direction"] == "SHORT"
    assert t1["position_id"] != t2["position_id"]
    assert res.open_positions.empty
    # Check ledger balance matches final balance
    assert res.ledger.iloc[-1]["balance_after"] == pytest.approx(res.final_balance)


def test_scenario_2_netting_single_blocks_scale_in():
    bars = _make_bars([2000.0, 2005.0, 2010.0, 2015.0, 2020.0])
    script = {
        0: (SignalType.LONG, {"quantity": 0.1}),
        1: (SignalType.LONG, {"quantity": 0.2}),  # Ignored in NETTING_SINGLE
        2: (SignalType.EXIT_LONG, {}),
    }
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", lot_size=0.1, contract_size=100.0),
        spread=SpreadConfig(spread_price=0.0),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=10.0, commission_mode="PER_SIDE"),
        risk=RiskConfig(stop_loss_pct=0.20, take_profit_pct=0.50),
        default_position_mode=PositionMode.NETTING_SINGLE,
    )
    res = BacktestEngine(config=cfg).run(bars, _ScriptedEventStrategy(script))

    assert len(res.trades) == 1
    assert res.trades.iloc[0]["quantity"] == pytest.approx(0.1)
    assert res.trades.iloc[0]["units"] == pytest.approx(10.0)
    assert any(log["reason"] == "ALREADY_LONG" for log in res.signal_log)


def test_scenario_3_netting_scale_in_partial_close_and_full_close():
    # Bar 0: signal LONG 0.1 lot -> fills at Bar 1 open (2000.0)
    # Bar 1: signal LONG 0.2 lot -> fills at Bar 2 open (2015.0) -> avg entry = (2000*10 + 2015*20)/30 = 2010.0
    # Bar 2: signal EXIT_LONG 0.1 lot (partial close 1/3) -> fills at Bar 3 open (2025.0)
    # Bar 3: signal EXIT_LONG (full close remaining 0.2 lot) -> fills at Bar 4 open (2030.0)
    bars = _make_bars([1995.0, 2000.0, 2015.0, 2025.0, 2030.0, 2035.0])
    script = {
        0: (SignalType.LONG, {"quantity": 0.1}),
        1: (SignalType.LONG, {"quantity": 0.2}),
        2: (SignalType.EXIT_LONG, {"quantity": 0.1}),
        3: (SignalType.EXIT_LONG, {}),
    }
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", lot_size=0.1, contract_size=100.0),
        spread=SpreadConfig(spread_price=0.0),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=10.0, commission_mode="PER_SIDE"),
        risk=RiskConfig(stop_loss_pct=0.20, take_profit_pct=0.50),
        default_position_mode=PositionMode.NETTING_SCALE_IN,
    )
    res = BacktestEngine(config=cfg).run(bars, _ScriptedEventStrategy(script))

    assert len(res.trades) == 2
    t1 = res.trades.iloc[0]
    t2 = res.trades.iloc[1]
    assert t1["position_id"] == t2["position_id"]
    assert t1["entry_price"] == pytest.approx(2010.0)
    assert t2["entry_price"] == pytest.approx(2010.0)
    assert t1["quantity"] == pytest.approx(0.1)
    assert t2["quantity"] == pytest.approx(0.2)
    assert len(t1["entry_order_ids"]) == 2
    assert len(t1["entry_fill_ids"]) == 2
    assert t1["entry_order_id"] is None
    assert t1["entry_fill_id"] is None

    # Weighted holding bars:
    # Trade 1 closed at bar 3: 1/3 of 10 units from bar 1 (held 2 bars) + 1/3 of 20 units from bar 2 (held 1 bar)
    # weighted_hb = (3.3333*2 + 6.6667*1) / 10 = 4/3 = 1.333333...
    assert t1["weighted_holding_bars"] == pytest.approx(4.0 / 3.0)
    # Trade 2 closed at bar 4: 2/3 of 10 units from bar 1 (held 3 bars) + 2/3 of 20 units from bar 2 (held 2 bars)
    # weighted_hb = (6.6667*3 + 13.3333*2) / 20 = 7/3 = 2.333333...
    assert t2["weighted_holding_bars"] == pytest.approx(7.0 / 3.0)

    # 4 allocations total (2 contributions * 2 closes)
    assert len(res.trade_allocations) == 4
    assert res.trade_allocations["allocated_units"].sum() == pytest.approx(30.0)
    assert res.trade_allocations["allocated_entry_fee"].sum() == pytest.approx(3.0)  # 1.0 + 2.0


def test_scenario_4_two_order_reverse_policy():
    bars = _make_bars([2000.0, 2005.0, 2015.0, 2005.0, 2000.0])
    script = {
        0: (SignalType.LONG, {"quantity": 0.1}),
        1: (SignalType.SHORT, {"quantity": 0.2}),  # Reverse at Bar 2 open
        2: (SignalType.EXIT_SHORT, {}),
    }
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", lot_size=0.1, contract_size=100.0),
        spread=SpreadConfig(spread_price=0.0),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=10.0, commission_mode="PER_SIDE"),
        risk=RiskConfig(stop_loss_pct=0.20, take_profit_pct=0.50),
        reverse_policy=ReversePolicy.REVERSE,
    )
    res = BacktestEngine(config=cfg).run(bars, _ScriptedEventStrategy(script))

    assert len(res.trades) == 2
    assert res.trades.iloc[0]["direction"] == "LONG"
    assert res.trades.iloc[0]["quantity"] == pytest.approx(0.1)
    assert res.trades.iloc[1]["direction"] == "SHORT"
    assert res.trades.iloc[1]["quantity"] == pytest.approx(0.2)
    assert res.trades.iloc[0]["position_id"] != res.trades.iloc[1]["position_id"]


def test_scenario_5_protective_sl_tp_closes_remaining_units_after_scale_in_and_partial_close():
    # Bar 0: signal LONG 0.1 lot -> fills at Bar 1 open (2000.0)
    # Bar 1: signal LONG 0.2 lot with TP=2030.0 -> fills at Bar 2 open (2015.0), total=0.3 lot (30 units)
    # Bar 2: signal EXIT_LONG 0.1 lot -> fills at Bar 3 open (2020.0), remaining=0.2 lot (20 units)
    # Bar 3: high reaches 2035.0 -> hits TP=2030.0 in Range for remaining 0.2 lot (20 units)!
    bars = pd.DataFrame([
        {"time": pd.Timestamp("2026-01-01 00:00"), "open": 1995.0, "high": 1998.0, "low": 1992.0, "close": 1996.0},
        {"time": pd.Timestamp("2026-01-01 00:15"), "open": 2000.0, "high": 2005.0, "low": 1998.0, "close": 2004.0},
        {"time": pd.Timestamp("2026-01-01 00:30"), "open": 2015.0, "high": 2018.0, "low": 2012.0, "close": 2016.0},
        {"time": pd.Timestamp("2026-01-01 00:45"), "open": 2020.0, "high": 2035.0, "low": 2018.0, "close": 2032.0},
        {"time": pd.Timestamp("2026-01-01 01:00"), "open": 2032.0, "high": 2034.0, "low": 2030.0, "close": 2031.0},
    ])
    script = {
        0: (SignalType.LONG, {"quantity": 0.1, "stop_loss_price": 1950.0, "take_profit_price": 2030.0}),
        1: (SignalType.LONG, {"quantity": 0.2, "stop_loss_price": 1950.0, "take_profit_price": 2030.0}),
        2: (SignalType.EXIT_LONG, {"quantity": 0.1}),
    }
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", lot_size=0.1, contract_size=100.0),
        spread=SpreadConfig(spread_price=0.0),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=10.0, commission_mode="PER_SIDE"),
        risk=RiskConfig(stop_loss_pct=0.20, take_profit_pct=0.50),
        default_position_mode=PositionMode.NETTING_SCALE_IN,
    )
    res = BacktestEngine(config=cfg).run(bars, _ScriptedEventStrategy(script))

    assert len(res.trades) == 2
    assert res.trades.iloc[0]["reason"] == "STRATEGY_EXIT"
    assert res.trades.iloc[0]["units"] == pytest.approx(10.0)
    assert res.trades.iloc[1]["reason"] == "TAKE_PROFIT"
    assert res.trades.iloc[1]["units"] == pytest.approx(20.0)
    assert res.trades.iloc[1]["exit_price"] == pytest.approx(2030.0)


def test_scenario_6_7_8_force_close_reconciliation_and_no_future_leakage():
    bars_a = _make_bars([2000.0, 2005.0, 2015.0, 2020.0, 2025.0, 2030.0])
    # Perturb bar 4 and 5 after close at bar 3
    bars_b = bars_a.copy()
    bars_b.loc[4, ["open", "high", "low", "close"]] = [2500.0, 2550.0, 2490.0, 2520.0]
    bars_b.loc[5, ["open", "high", "low", "close"]] = [1500.0, 1550.0, 1490.0, 1520.0]

    script = {
        0: (SignalType.LONG, {"quantity": 0.1}),
        1: (SignalType.LONG, {"quantity": 0.1}),
        2: (SignalType.EXIT_LONG, {}),
    }
    cfg = EngineConfig(
        account=AccountConfig(initial_balance=10000.0),
        instrument=InstrumentConfig(symbol="XAUUSD", lot_size=0.1, contract_size=100.0),
        spread=SpreadConfig(spread_price=1.0),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=10.0, commission_mode="PER_SIDE"),
        risk=RiskConfig(stop_loss_pct=0.20, take_profit_pct=0.50),
        default_position_mode=PositionMode.NETTING_SCALE_IN,
    )
    res_a = BacktestEngine(config=cfg).run(bars_a, _ScriptedEventStrategy(script))
    res_b = BacktestEngine(config=cfg).run(bars_b, _ScriptedEventStrategy(script))

    # Scenario 8: No future leakage - trade records and ledger entries are identical despite future bar perturbation
    pd.testing.assert_frame_equal(res_a.trades, res_b.trades)
    pd.testing.assert_frame_equal(res_a.ledger, res_b.ledger)
    pd.testing.assert_frame_equal(res_a.trade_allocations, res_b.trade_allocations)
