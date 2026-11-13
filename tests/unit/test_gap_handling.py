import pandas as pd
import pytest

from quant_backtester.execution.slippage import FixedSlippageModel
from quant_backtester.execution.standard import StandardExecutionModel
from quant_backtester.models.order import ExecutionPhase, OrderOrigin
from quant_backtester.models.quote import QuoteBar
from quant_backtester.portfolio.position import Position, PositionSide


def test_long_and_short_protective_sl_and_tp_normal_vs_gap():
    exec_model = StandardExecutionModel(slippage_model=FixedSlippageModel(0.20))
    ts = pd.Timestamp("2026-01-01 00:15")

    # 1. Long Position (Entry = 2000, SL = 1990, TP = 2020)
    pos_long = Position(symbol="XAUUSD")
    pos_long.open(
        PositionSide.LONG,
        price=2000.0,
        units=1.0,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        stop_loss_price=1990.0,
        take_profit_price=2020.0,
        opened_bar_index=0,
        opened_phase=ExecutionPhase.OPEN,
    )

    # 1a. Long SL Gap at Open: Bid Open = 1985.0 <= 1990.0 -> Base = 1985.0, Fill = 1985.0 - 0.20 = 1984.80 (with slippage)
    qb_gap_down = QuoteBar("XAUUSD", ts, 1985.0, 1992.0, 1980.0, 1988.0, 1985.2, 1992.2, 1980.2, 1988.2)
    prop_sl_gap = exec_model.propose_protective_open(pos_long, qb_gap_down, bar_index=1)
    assert prop_sl_gap is not None
    assert prop_sl_gap.origin == OrderOrigin.PROTECTIVE_SL
    assert prop_sl_gap.base_price == pytest.approx(1985.0)
    assert prop_sl_gap.fill_price == pytest.approx(1984.80)

    # 1b. Long TP Gap at Open: Bid Open = 2025.0 >= 2020.0 -> Base = 2025.0, Fill = 2025.0 (NO slippage on TP)
    qb_gap_up = QuoteBar("XAUUSD", ts, 2025.0, 2030.0, 2020.0, 2028.0, 2025.2, 2030.2, 2020.2, 2028.2)
    prop_tp_gap = exec_model.propose_protective_open(pos_long, qb_gap_up, bar_index=1)
    assert prop_tp_gap is not None
    assert prop_tp_gap.origin == OrderOrigin.PROTECTIVE_TP
    assert prop_tp_gap.fill_price == pytest.approx(2025.0)
    assert prop_tp_gap.slippage == 0.0

    # 2. Short Position (Entry = 2000, SL = 2010, TP = 1980) -> evaluated against ASK prices!
    pos_short = Position(symbol="XAUUSD")
    pos_short.open(
        PositionSide.SHORT,
        price=2000.0,
        units=1.0,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        stop_loss_price=2010.0,
        take_profit_price=1980.0,
        opened_bar_index=0,
        opened_phase=ExecutionPhase.OPEN,
    )

    # 2a. Short SL normal in Range: Ask High = 2012.0 >= 2010.0 -> Base = 2010.0, Fill = 2010.0 + 0.20 = 2010.20
    qb_short_sl = QuoteBar("XAUUSD", ts, 2000.0, 2011.8, 1995.0, 2005.0, 2000.2, 2012.0, 1995.2, 2005.2)
    prop_s_sl = exec_model.propose_protective_range(pos_short, qb_short_sl, bar_index=1)
    assert prop_s_sl is not None
    assert prop_s_sl.origin == OrderOrigin.PROTECTIVE_SL
    assert prop_s_sl.base_price == pytest.approx(2010.0)
    assert prop_s_sl.fill_price == pytest.approx(2010.20)

    # 2b. Short TP Gap at Open: Ask Open = 1975.0 <= 1980.0 -> Base = 1975.0, Fill = 1975.0 (NO slippage)
    qb_short_tp_gap = QuoteBar("XAUUSD", ts, 1974.8, 1980.0, 1970.0, 1976.0, 1975.0, 1980.2, 1970.2, 1976.2)
    prop_s_tp = exec_model.propose_protective_open(pos_short, qb_short_tp_gap, bar_index=1)
    assert prop_s_tp is not None
    assert prop_s_tp.origin == OrderOrigin.PROTECTIVE_TP
    assert prop_s_tp.fill_price == pytest.approx(1975.0)
    assert prop_s_tp.slippage == 0.0

