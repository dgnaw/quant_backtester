import numpy as np
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal
from quant_backtester.portfolio.position import PositionSide, PositionView
from quant_backtester.strategies.adapters import frame_to_signals
from quant_backtester.strategies.base import Strategy, StrategyMode
from quant_backtester.strategies.context import StrategyContext
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy


def _collect_event_driven_signals(df: pd.DataFrame, strategy: Strategy, symbol: str = "XAUUSD") -> list[Signal | None]:
    strategy.reset()
    out: list[Signal | None] = []
    pos = PositionView(symbol=symbol, side=PositionSide.FLAT, units=0.0, entry_price=0.0)
    for idx, row in df.iterrows():
        ctx = StrategyContext(
            time=pd.Timestamp(row["time"]),
            bar_index=int(idx),
            symbol=symbol,
            position=pos,
            data=df,
        )
        sig = strategy.on_bar(row, ctx)
        out.append(sig if not isinstance(sig, list) else (sig[0] if sig else None))
    return out


def test_mean_reversion_vectorized_and_event_driven_equivalence():
    np.random.seed(42)
    n = 80
    closes = 2000.0 + np.cumsum(np.random.randn(n) * 6.0)
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 2.0) + 0.5
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 2.0) - 0.5

    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=n, freq="15min"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })

    cfg = BacktestConfig(stop_loss_pct=0.01, take_profit_pct=0.02, fee_rate_per_side=0.00015)

    strat_vec = MeanReversionStrategy(window=15, std_multiplier=1.5)
    res_vec = BacktestEngine(cfg).run(df, strat_vec, run_mode=StrategyMode.VECTORIZED)

    strat_evt = MeanReversionStrategy(window=15, std_multiplier=1.5)
    res_evt = BacktestEngine(cfg).run(df, strat_evt, run_mode=StrategyMode.EVENT_DRIVEN)

    assert len(res_vec.trades) == len(res_evt.trades)
    pd.testing.assert_frame_equal(res_vec.trades, res_evt.trades)
    pd.testing.assert_frame_equal(res_vec.equity, res_evt.equity)
    assert res_vec.final_balance == pytest.approx(res_evt.final_balance, rel=1e-9)
    assert res_vec.final_equity == pytest.approx(res_evt.final_equity, rel=1e-9)


def test_causality_harness_future_perturbation():
    """
    9.1 Future Perturbation Test:
    Thay đổi dữ liệu từ cutoff + 1 trở đi không được làm thay đổi bất kỳ Signal nào từ 0 đến cutoff.
    Kiểm tra cho cả chế độ Vectorized và Event-driven.
    """
    np.random.seed(7)
    n = 50
    cutoff = 25
    closes = 2000.0 + np.sin(np.arange(n) / 3.0) * 15.0
    df_orig = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=n, freq="15min"),
        "open": closes - 1.0,
        "high": closes + 3.0,
        "low": closes - 3.0,
        "close": closes,
    })

    df_perturbed = df_orig.copy(deep=True)
    # Thay đổi mạnh dữ liệu từ cutoff + 1 trở đi
    df_perturbed.loc[cutoff + 1 :, "open"] *= 3.5
    df_perturbed.loc[cutoff + 1 :, "high"] *= 4.0
    df_perturbed.loc[cutoff + 1 :, "low"] *= 0.2
    df_perturbed.loc[cutoff + 1 :, "close"] *= 3.5

    strat = MeanReversionStrategy(window=10, std_multiplier=1.5)

    # 1. Kiểm tra Vectorized signals [0 : cutoff]
    vec_orig = frame_to_signals(strat.generate_signals(df_orig), symbol="XAUUSD")
    vec_mod = frame_to_signals(strat.generate_signals(df_perturbed), symbol="XAUUSD")
    assert vec_orig[: cutoff + 1] == vec_mod[: cutoff + 1]

    # 2. Kiểm tra Event-driven signals [0 : cutoff]
    evt_orig = _collect_event_driven_signals(df_orig, strat, symbol="XAUUSD")
    evt_mod = _collect_event_driven_signals(df_perturbed, strat, symbol="XAUUSD")
    assert evt_orig[: cutoff + 1] == evt_mod[: cutoff + 1]

