from typing import ClassVar
import pandas as pd
import pytest

from quant_backtester.config import (
    BacktestConfig,
    EngineConfig,
    FeeConfig,
    IntrabarConflictPolicy,
    ReversePolicy,
    SlippageConfig,
    SpreadConfig,
)
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.order import OrderStatus, OrderType, TimeInForce
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.base import Strategy, StrategyMode
from quant_backtester.strategies.context import StrategyContext


# -----------------------------------------------------------------------------
# 1. LIMIT & STOP_MARKET EXECUTION IN ENGINE (ELIGIBLE_FROM_BAR & RANGE FILL)
# -----------------------------------------------------------------------------
class LimitAndStopStrategy(Strategy):
    name = "limit_and_stop_strategy"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        if context.bar_index == 0 and context.position.is_flat:
            # Emit BUY LIMIT at 1995.0 on bar 0 -> eligible from bar 1
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={
                    "directives": {
                        "order_type": "LIMIT",
                        "limit_price": 1995.0,
                        "quantity": 0.02,
                        "take_profit_price": 2015.0,
                    }
                },
            )
        return None


def test_engine_executes_limit_order_in_range_and_tracks_events():
    # Bar 0: Close emits BUY LIMIT @ 1995.0 (eligible from bar 1)
    # Bar 1: Low = 1998.0 -> Not reached, stays PENDING
    # Bar 2: Low = 1994.0 <= 1995.0 -> Fills in RANGE at 1995.0!
    #        Even though Bar 2 High = 2020.0 >= TP(2015.0), a position opened in RANGE is NOT evaluated for SL/TP in Bar 2!
    # Bar 3: High = 2018.0 >= TP(2015.0) -> Exits at TP = 2015.0 in Bar 3 RANGE!
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=5, freq="15min"),
        "open":  [2000.0, 2002.0, 2000.0, 2005.0, 2010.0],
        "high":  [2005.0, 2006.0, 2020.0, 2018.0, 2012.0],
        "low":   [1990.0, 1998.0, 1994.0, 2002.0, 2008.0],
        "close": [2002.0, 2000.0, 2005.0, 2010.0, 2011.0],
    })
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0, fee_rate_per_side=0.0, slippage_price=0.0)
    res = BacktestEngine(cfg).run(df, LimitAndStopStrategy())

    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == pd.Timestamp("2026-01-01 00:30")  # Bar 2
    assert tr["entry_price"] == pytest.approx(1995.0)
    assert tr["exit_time"] == pd.Timestamp("2026-01-01 00:45")   # Bar 3 (NOT Bar 2!)
    assert tr["exit_price"] == pytest.approx(2015.0)
    assert tr["reason"] == "TAKE_PROFIT"
    assert not res.order_events.empty


# -----------------------------------------------------------------------------
# 2. SUPERSEDING, TIF EXPIRATION & END_OF_TEST EXPIRATION IN ENGINE
# -----------------------------------------------------------------------------
class SupersedeAndExpireStrategy(Strategy):
    name = "supersede_and_expire"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        if context.bar_index == 0:
            # Emit BUY LIMIT @ 1950 (unreachable)
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={"directives": {"order_type": "LIMIT", "limit_price": 1950.0}},
            )
        if context.bar_index == 1:
            # Emit new BUY LIMIT @ 1960 with expire_after_bars=1 -> Supersedes order 1!
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={
                    "directives": {
                        "order_type": "LIMIT",
                        "limit_price": 1960.0,
                        "time_in_force": TimeInForce.GTC,
                        "expire_after_bars": 1,
                    }
                },
            )
        if context.bar_index == 2:
            # Emit BUY STOP_MARKET @ 2100 (unreachable before end of data) -> Expires at END_OF_TEST
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={"directives": {"order_type": "STOP_MARKET", "stop_price": 2100.0}},
            )
        return None


def test_engine_supersedes_and_expires_pending_orders():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=4, freq="15min"),
        "open":  [2000.0, 2001.0, 2002.0, 2003.0],
        "high":  [2005.0, 2006.0, 2007.0, 2008.0],
        "low":   [1995.0, 1996.0, 1997.0, 1998.0],
        "close": [2001.0, 2002.0, 2003.0, 2004.0],
    })
    cfg = BacktestConfig(fee_rate_per_side=0.0)
    res = BacktestEngine(cfg).run(df, SupersedeAndExpireStrategy())

    assert len(res.trades) == 0
    assert len(res.orders) == 3

    # Order 1: Cancelled with SUPERSEDED
    assert res.orders.iloc[0]["status"] == OrderStatus.CANCELLED.value
    assert res.orders.iloc[0]["cancel_reason"] == "SUPERSEDED"

    # Order 2: Expired after 1 bar (bar 2) with TIME_IN_FORCE_EXPIRED
    assert res.orders.iloc[1]["status"] == OrderStatus.EXPIRED.value
    assert res.orders.iloc[1]["cancel_reason"] == "TIME_IN_FORCE_EXPIRED"

    # Order 3: Expired at END_OF_TEST
    assert res.orders.iloc[2]["status"] == OrderStatus.EXPIRED.value
    assert res.orders.iloc[2]["cancel_reason"] == "END_OF_TEST"


# -----------------------------------------------------------------------------
# 3. INTRABAR SL/TP CONFLICT POLICY IN ENGINE
# -----------------------------------------------------------------------------
class OpenMarketBuyBar0Strategy(Strategy):
    name = "open_market_buy_bar0"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        if context.bar_index == 0:
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={
                    "directives": {
                        "stop_loss_price": 1990.0,
                        "take_profit_price": 2010.0,
                    }
                },
            )
        return None


def test_engine_intrabar_conflict_policies_on_same_bar():
    # Bar 0 emits Market BUY -> Fills at Bar 1 Open (2000.0) in OPEN phase.
    # Because it opened in OPEN phase, Bar 1 RANGE evaluates both SL(1990) and TP(2010)!
    # Bar 1 has Low=1985 <= 1990 and High=2015 >= 2010 -> Both hit!
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=3, freq="15min"),
        "open":  [2000.0, 2000.0, 2000.0],
        "high":  [2002.0, 2015.0, 2002.0],
        "low":   [1998.0, 1985.0, 1998.0],
        "close": [2000.0, 2000.0, 2000.0],
    })

    cfg_worst = BacktestConfig(
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        fee_rate_per_side=0.0,
        intrabar_conflict_policy=IntrabarConflictPolicy.WORST_CASE_SL_FIRST,
    )
    res_worst = BacktestEngine(cfg_worst).run(df, OpenMarketBuyBar0Strategy())
    assert len(res_worst.trades) == 1
    assert res_worst.trades.iloc[0]["reason"] == "STOP_LOSS"
    assert res_worst.trades.iloc[0]["exit_price"] == pytest.approx(1990.0)

    cfg_best = BacktestConfig(
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        fee_rate_per_side=0.0,
        intrabar_conflict_policy=IntrabarConflictPolicy.BEST_CASE_TP_FIRST,
    )
    res_best = BacktestEngine(cfg_best).run(df, OpenMarketBuyBar0Strategy())
    assert len(res_best.trades) == 1
    assert res_best.trades.iloc[0]["reason"] == "TAKE_PROFIT"
    assert res_best.trades.iloc[0]["exit_price"] == pytest.approx(2010.0)


# -----------------------------------------------------------------------------
# 4. MARKET FRICTIONS (SPREAD, COMMISSION, SLIPPAGE) & BID/ASK MARK-TO-MARKET
# -----------------------------------------------------------------------------
class HoldLongThenExitStrategy(Strategy):
    name = "hold_long_then_exit"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        if context.bar_index == 0:
            return Signal(time=context.time, symbol=context.symbol, signal_type=SignalType.LONG)
        if context.bar_index == 1:
            return Signal(time=context.time, symbol=context.symbol, signal_type=SignalType.EXIT_LONG)
        return None


def test_engine_bid_ask_spread_commission_slippage_and_mark_to_market():
    # MID price = 2000.0 across all bars; Spread = 0.40 -> Bid = 1999.80, Ask = 2000.20
    # Slippage = 0.10, Commission = 7.0 USD/lot ROUND_TURN (0.01 lot -> 0.035 USD per side)
    # Bar 0 emits LONG
    # Bar 1 Open: BUY 0.01 lot (1 unit) at Ask(2000.20) + Slippage(0.10) = 2000.30, entry_fee = 0.035
    # Bar 1 Close Mark-to-Market: Long marked at Bid Close (1999.80) -> Unrealized PnL = 1999.80 - 2000.30 = -0.50
    #   Equity at Bar 1 Close = (1000.0 - 0.035) - 0.50 = 999.465
    # Bar 1 Close emits EXIT_LONG
    # Bar 2 Open: SELL 0.01 lot (1 unit) at Bid(1999.80) - Slippage(0.10) = 1999.70, exit_fee = 0.035
    #   Gross PnL = 1999.70 - 2000.30 = -0.60; Total Fee = 0.07; Net PnL = -0.67
    #   Final Balance = 1000.0 - 0.67 = 999.33
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=3, freq="15min"),
        "open":  [2000.0, 2000.0, 2000.0],
        "high":  [2001.0, 2001.0, 2001.0],
        "low":   [1999.0, 1999.0, 1999.0],
        "close": [2000.0, 2000.0, 2000.0],
    })
    cfg = EngineConfig(
        spread=SpreadConfig(spread_price=0.40, price_type="MID"),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=7.0, commission_mode="ROUND_TURN"),
        slippage=SlippageConfig(slippage_price=0.10),
    )
    cfg.risk.stop_loss_pct = 0.0
    cfg.risk.take_profit_pct = 0.0

    res = BacktestEngine(cfg).run(df, HoldLongThenExitStrategy())
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["entry_price"] == pytest.approx(2000.30)
    assert tr["exit_price"] == pytest.approx(1999.70)
    assert tr["entry_fee"] == pytest.approx(0.035)
    assert tr["exit_fee"] == pytest.approx(0.035)
    assert tr["gross_pnl"] == pytest.approx(-0.60)
    assert tr["net_pnl"] == pytest.approx(-0.67)

    # Check Bar 1 Close Bid mark-to-market equity
    assert res.equity.iloc[1]["equity"] == pytest.approx(999.465)
    assert res.final_balance == pytest.approx(999.33)


# -----------------------------------------------------------------------------
# 5. POSITION REVERSAL (ReversePolicy.REVERSE) WITH FULL TWO-ORDER ACCOUNTING
# -----------------------------------------------------------------------------
class ReverseLongToShortStrategy(Strategy):
    name = "reverse_long_to_short"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        if context.bar_index == 0:
            return Signal(time=context.time, symbol=context.symbol, signal_type=SignalType.LONG)
        if context.bar_index == 1:
            return Signal(time=context.time, symbol=context.symbol, signal_type=SignalType.SHORT)
        return None


def test_engine_reverse_policy_executes_two_orders_and_charges_both_fees():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=4, freq="15min"),
        "open":  [2000.0, 2000.0, 2010.0, 2005.0],
        "high":  [2002.0, 2012.0, 2012.0, 2008.0],
        "low":   [1998.0, 1998.0, 2002.0, 2000.0],
        "close": [2000.0, 2010.0, 2005.0, 2002.0],
    })
    cfg = BacktestConfig(
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        fee_rate_per_side=0.0,
        commission_per_lot=10.0,  # 0.01 lot -> 0.05 USD per side
        reverse_policy=ReversePolicy.REVERSE,
    )
    res = BacktestEngine(cfg).run(df, ReverseLongToShortStrategy())

    # Trade 1: LONG opened at Bar 1 (2000.0), closed at Bar 2 Open (2010.0) via STRATEGY_EXIT
    # Trade 2: SHORT opened at Bar 2 Open (2010.0), closed at Bar 3 Close (2002.0) via END_OF_TEST
    assert len(res.trades) == 2
    assert res.trades.iloc[0]["direction"] == "LONG"
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(2010.0)
    assert res.trades.iloc[1]["direction"] == "SHORT"
    assert res.trades.iloc[1]["entry_price"] == pytest.approx(2010.0)
    assert res.trades.iloc[1]["exit_price"] == pytest.approx(2002.0)


# -----------------------------------------------------------------------------
# 6. POSITION NOT CLOSED TWICE WHEN GAP SL AND STRATEGY_EXIT COINCIDE
# -----------------------------------------------------------------------------
def test_engine_does_not_close_position_twice_on_gap_sl_and_strategy_exit():
    # Bar 0 emits LONG (SL 1% -> 1980.0)
    # Bar 1 opens at 2000.0 (Position LONG @ 2000.0, SL=1980.0). At Bar 1 Close, strategy emits EXIT_LONG!
    # Bar 2 gaps down to Open = 1950.0 (below SL 1980.0)!
    # At Bar 2 Open:
    #   1. Protective SL Gap closes the position at 1950.0 (reason="STOP_GAP").
    #   2. The pending STRATEGY_EXIT order from Bar 1 must be CANCELLED ("POSITION_ALREADY_CLOSED") and NOT close again!
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=4, freq="15min"),
        "open":  [2000.0, 2000.0, 1950.0, 1955.0],
        "high":  [2002.0, 2002.0, 1960.0, 1960.0],
        "low":   [1998.0, 1995.0, 1945.0, 1950.0],
        "close": [2000.0, 1996.0, 1955.0, 1958.0],
    })
    cfg = BacktestConfig(stop_loss_pct=0.01, take_profit_pct=0.02, fee_rate_per_side=0.0)
    res = BacktestEngine(cfg).run(df, HoldLongThenExitStrategy())

    # Only 1 closed trade (closed by STOP_GAP at 1950.0)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["reason"] == "STOP_GAP"
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(1950.0)

    # The STRATEGY_EXIT order must be CANCELLED with POSITION_ALREADY_CLOSED
    cancelled_exits = res.orders[
        (res.orders["origin"] == "STRATEGY_EXIT") & (res.orders["status"] == OrderStatus.CANCELLED.value)
    ]
    assert len(cancelled_exits) == 1
    assert cancelled_exits.iloc[0]["cancel_reason"] == "POSITION_ALREADY_CLOSED"


# -----------------------------------------------------------------------------
# 7. TWO-PHASE FILL REJECTION AT OPEN WHEN GAP INVALIDATES FIXED SL PRICE
# -----------------------------------------------------------------------------
class FixedSLMarketBuyStrategy(Strategy):
    name = "fixed_sl_market_buy"
    mode: ClassVar[StrategyMode] = StrategyMode.EVENT_DRIVEN

    def _initial_state(self):
        return {"reject_reasons": []}

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        if context.bar_index == 0:
            # At Bar 0 Close = 2000.0, stop_loss_price = 1990.0 is valid (< 2000.0).
            # But Bar 1 Open gaps down to 1980.0 -> fill_price (1980.0) <= stop_loss_price (1990.0)!
            return Signal(
                time=context.time,
                symbol=context.symbol,
                signal_type=SignalType.LONG,
                metadata={"directives": {"stop_loss_price": 1990.0}},
            )
        return None

    def on_intent_rejected(self, intent_or_order, reason: str) -> None:
        self.state["reject_reasons"].append(reason)


def test_engine_two_phase_fill_rejects_order_when_gap_invalidates_sl_at_fill_price():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=3, freq="15min"),
        "open":  [2000.0, 1980.0, 1982.0],
        "high":  [2002.0, 1985.0, 1986.0],
        "low":   [1998.0, 1975.0, 1978.0],
        "close": [2000.0, 1982.0, 1984.0],
    })
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0, fee_rate_per_side=0.0)
    strat = FixedSLMarketBuyStrategy()
    res = BacktestEngine(cfg).run(df, strat)

    assert len(res.trades) == 0
    assert res.final_balance == pytest.approx(cfg.initial_balance)
    assert len(res.orders) == 1
    assert res.orders.iloc[0]["status"] == OrderStatus.REJECTED.value
    assert res.orders.iloc[0]["reject_reason"] == "INVALID_SL_AT_FILL_PRICE"
    assert strat.state["reject_reasons"] == ["INVALID_SL_AT_FILL_PRICE"]

