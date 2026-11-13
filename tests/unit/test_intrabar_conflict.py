import pandas as pd
import pytest

from quant_backtester.config import IntrabarConflictPolicy
from quant_backtester.execution.standard import StandardExecutionModel
from quant_backtester.models.order import ExecutionPhase, OrderOrigin
from quant_backtester.models.quote import QuoteBar
from quant_backtester.portfolio.position import Position, PositionSide


def test_intrabar_conflict_worst_case_vs_best_case_and_metadata():
    # Long Entry = 100, SL = 95, TP = 110; Bar has Bid Low = 94 (<= 95) and Bid High = 112 (>= 110)
    ts = pd.Timestamp("2026-01-01 00:15")
    qb = QuoteBar("XAUUSD", ts, 100.0, 112.0, 94.0, 105.0, 100.0, 112.0, 94.0, 105.0)

    pos = Position(symbol="XAUUSD")
    pos.open(
        PositionSide.LONG,
        price=100.0,
        units=1.0,
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        stop_loss_price=95.0,
        take_profit_price=110.0,
        opened_bar_index=0,
        opened_phase=ExecutionPhase.OPEN,
    )

    # 1. Default WORST_CASE_SL_FIRST -> triggers SL at 95.0 and marks ambiguous_sl_tp=True
    exec_worst = StandardExecutionModel(conflict_policy=IntrabarConflictPolicy.WORST_CASE_SL_FIRST)
    prop_worst = exec_worst.propose_protective_range(pos, qb, bar_index=1)
    assert prop_worst is not None
    assert prop_worst.origin == OrderOrigin.PROTECTIVE_SL
    assert prop_worst.fill_price == pytest.approx(95.0)
    assert prop_worst.metadata.get("ambiguous_sl_tp") is True

    # 2. BEST_CASE_TP_FIRST -> triggers TP at 110.0 and marks ambiguous_sl_tp=True
    exec_best = StandardExecutionModel(conflict_policy=IntrabarConflictPolicy.BEST_CASE_TP_FIRST)
    prop_best = exec_best.propose_protective_range(pos, qb, bar_index=1)
    assert prop_best is not None
    assert prop_best.origin == OrderOrigin.PROTECTIVE_TP
    assert prop_best.fill_price == pytest.approx(110.0)
    assert prop_best.metadata.get("ambiguous_sl_tp") is True

