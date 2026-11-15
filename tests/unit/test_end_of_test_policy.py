import pandas as pd
import pytest

from quant_backtester.config import (
    AccountConfig,
    EngineConfig,
    FeeConfig,
    InstrumentConfig,
    RiskConfig,
    SlippageConfig,
    SpreadConfig,
)
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.portfolio.account import Account
from quant_backtester.portfolio.ledger import LEDGER_COLUMNS
from quant_backtester.portfolio.portfolio import (
    OPEN_POSITION_COLUMNS,
    TRADE_ALLOCATION_COLUMNS,
    Portfolio,
)
from quant_backtester.portfolio.position import PositionSide
from quant_backtester.strategies.base import Strategy, StrategyMode


def _make_bars(n: int = 5) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2026-01-01 00:00")
    prices = [2000.0, 2005.0, 2010.0, 2015.0, 2020.0]
    for i in range(n):
        p = prices[i % len(prices)]
        rows.append({
            "time": t0 + pd.Timedelta(minutes=15 * i),
            "open": p,
            "high": p + 2.0,
            "low": p - 2.0,
            "close": p + 1.0,
        })
    return pd.DataFrame(rows)


class _HoldSingleSideStrategy(Strategy):
    mode = StrategyMode.EVENT_DRIVEN

    def __init__(self, side_signal: SignalType):
        super().__init__(side_signal=side_signal.value)
        self.side_signal = side_signal

    def on_bar(self, bar, ctx):
        if ctx.bar_index == 0:
            return Signal(
                time=ctx.time,
                symbol=ctx.symbol,
                signal_type=self.side_signal,
                metadata={"directives": {"quantity": 0.1}},
            )
        return None


def _base_config(force_close: bool) -> EngineConfig:
    return EngineConfig(
        account=AccountConfig(initial_balance=10000.0, base_currency="USD"),
        instrument=InstrumentConfig(symbol="XAUUSD", lot_size=0.1, contract_size=100.0),
        spread=SpreadConfig(spread_price=2.0, price_type="MID"),
        fee=FeeConfig(fee_rate_per_side=0.0, commission_per_lot=10.0, commission_mode="PER_SIDE"),
        slippage=SlippageConfig(slippage_price=0.5),
        risk=RiskConfig(stop_loss_pct=0.20, take_profit_pct=0.50),
        force_close_at_end=force_close,
    )


def test_force_close_at_end_true_for_long_and_short():
    bars = _make_bars(4)

    # 1. LONG with force_close_at_end = True
    cfg_long = _base_config(force_close=True)
    res_long = BacktestEngine(config=cfg_long).run(bars, _HoldSingleSideStrategy(SignalType.LONG))
    assert len(res_long.trades) == 1
    assert res_long.trades.iloc[0]["reason"] == "END_OF_TEST"
    assert res_long.open_positions.empty
    assert list(res_long.open_positions.columns) == OPEN_POSITION_COLUMNS
    assert res_long.final_equity == pytest.approx(res_long.final_balance)

    # 2. SHORT with force_close_at_end = True
    cfg_short = _base_config(force_close=True)
    res_short = BacktestEngine(config=cfg_short).run(bars, _HoldSingleSideStrategy(SignalType.SHORT))
    assert len(res_short.trades) == 1
    assert res_short.trades.iloc[0]["reason"] == "END_OF_TEST"
    assert res_short.open_positions.empty
    assert res_short.final_equity == pytest.approx(res_short.final_balance)


def test_force_close_at_end_false_for_long_and_short_and_result_exports():
    bars = _make_bars(4)
    # Last bar (index 3): close = 2016.0, spread = 2.0 (MID) -> bid_close = 2015.0, ask_close = 2017.0

    # 3. LONG with force_close_at_end = False
    cfg_long = _base_config(force_close=False)
    res_long = BacktestEngine(config=cfg_long).run(bars, _HoldSingleSideStrategy(SignalType.LONG))
    assert res_long.trades.empty
    assert len(res_long.open_positions) == 1
    op_long = res_long.open_positions.iloc[0]
    assert op_long["side"] == "LONG"
    assert op_long["mark_price"] == pytest.approx(2015.0)
    assert res_long.final_equity == pytest.approx(res_long.final_balance + op_long["unrealized_pnl"])

    # 4. SHORT with force_close_at_end = False
    cfg_short = _base_config(force_close=False)
    res_short = BacktestEngine(config=cfg_short).run(bars, _HoldSingleSideStrategy(SignalType.SHORT))
    assert res_short.trades.empty
    assert len(res_short.open_positions) == 1
    op_short = res_short.open_positions.iloc[0]
    assert op_short["side"] == "SHORT"
    assert op_short["mark_price"] == pytest.approx(2017.0)
    assert res_short.final_equity == pytest.approx(res_short.final_balance + op_short["unrealized_pnl"])

    # 5, 6, 7, 8. DataFrame schemas and dictionary __getitem__ access
    assert list(res_long.ledger.columns) == LEDGER_COLUMNS
    assert list(res_long.open_positions.columns) == OPEN_POSITION_COLUMNS
    assert list(res_long.trade_allocations.columns) == TRADE_ALLOCATION_COLUMNS
    assert "ledger" in res_long
    assert "open_positions" in res_long
    assert "trade_allocations" in res_long
    assert res_long["ledger"] is res_long.ledger
    assert res_long["open_positions"] is res_long.open_positions
    assert res_long["trade_allocations"] is res_long.trade_allocations


def test_open_positions_to_dataframe_requires_quotebar_for_mark_prices():
    acc = Account(initial_balance=10000.0)
    pf = Portfolio(account=acc, instrument_config=InstrumentConfig(symbol="XAUUSD", contract_size=100.0))
    pos = pf.get_position("XAUUSD")
    pos.open(side=PositionSide.LONG, price=2000.0, quantity=0.1, timestamp=pd.Timestamp("2026-01-01"))
    pf.active_positions["XAUUSD"] = pos

    with pytest.raises(TypeError, match="QuoteBar"):
        pf.open_positions_to_dataframe({"XAUUSD": 2010.0})  # type: ignore[dict-item]
