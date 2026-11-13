from pathlib import Path
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig
from quant_backtester.data import load_ohlc_csv
from quant_backtester.engine import BacktestEngine
from quant_backtester.execution.fee import PercentageFeeModel
from quant_backtester.execution.immediate import ImmediateExecutionModel
from quant_backtester.execution.slippage import FixedSlippageModel
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy


def _sample_df() -> pd.DataFrame:
    sample_path = Path(__file__).resolve().parents[2] / "examples" / "sample_ohlc.csv"
    return load_ohlc_csv(sample_path)


def _legacy_engine(cfg: BacktestConfig) -> BacktestEngine:
    exec_model = ImmediateExecutionModel(
        fee_model=PercentageFeeModel(cfg.fee.fee_rate_per_side),
        slippage_model=FixedSlippageModel(cfg.slippage.slippage_price),
    )
    return BacktestEngine(config=cfg, execution_model=exec_model)


def test_golden_baseline_1_no_sl_tp():
    df = _sample_df()
    cfg = BacktestConfig(stop_loss_pct=0.0, take_profit_pct=0.0, fee_rate_per_side=0.0, slippage_price=0.0)
    res = _legacy_engine(cfg).run(df, MeanReversionStrategy(20, 2.0))

    assert len(res.equity) == len(df)
    assert len(res.trades) > 0
    assert (res.trades["entry_fee"] == 0.0).all()
    assert (res.trades["exit_fee"] == 0.0).all()
    total_pnl = float(res.trades["net_pnl"].sum())
    assert res.final_balance == pytest.approx(cfg.initial_balance + total_pnl, rel=1e-9)


def test_golden_baseline_2_sl_1pct_tp_2pct_and_default_fee():
    df = _sample_df()
    cfg = BacktestConfig(stop_loss_pct=0.01, take_profit_pct=0.02, fee_rate_per_side=0.00015, slippage_price=0.0)
    res1 = _legacy_engine(cfg).run(df, MeanReversionStrategy(20, 2.0))
    res2 = BacktestEngine(cfg).run(df, MeanReversionStrategy(20, 2.0))

    # Deterministic & identical between legacy ImmediateExecutionModel and default ZeroSpread/ZeroSlippage StandardExecutionModel
    assert len(res1.trades) == len(res2.trades)
    assert res1.final_balance == pytest.approx(res2.final_balance, rel=1e-9)
    assert res1.final_equity == pytest.approx(res2.final_equity, rel=1e-9)
    total_comm = float((res1.trades["entry_fee"] + res1.trades["exit_fee"]).sum())
    assert total_comm > 0.0


def test_golden_baseline_3_gap_sl_and_tp():
    class BuyBar0Strategy(Strategy):
        name = "buy_bar_0"

        def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
            out = market_data.copy()
            out["entry_signal"] = 0
            out["exit_long_signal"] = False
            out["exit_short_signal"] = False
            out.loc[0, "entry_signal"] = 1
            return StrategySignals(out)

    # Bar 0 emits LONG -> Bar 1 opens at 2000.0 (SL = 1980.0) -> Bar 2 gaps down to Open = 1950.0
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=4, freq="15min"),
        "open":  [2000.0, 2000.0, 1950.0, 1955.0],
        "high":  [2005.0, 2005.0, 1960.0, 1960.0],
        "low":   [1995.0, 1995.0, 1945.0, 1950.0],
        "close": [2000.0, 1998.0, 1955.0, 1958.0],
    })
    cfg = BacktestConfig(stop_loss_pct=0.01, take_profit_pct=0.02, fee_rate_per_side=0.0, slippage_price=0.0)
    res = BacktestEngine(cfg).run(df, BuyBar0Strategy())
    assert len(res.trades) == 1
    # Gap down below 1980 must fill at Open 1950.0, not 1980.0
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(1950.0)

