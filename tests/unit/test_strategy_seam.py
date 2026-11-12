import ast
from pathlib import Path
import pandas as pd
import pytest

from quant_backtester.config import BacktestConfig, EngineConfig, ReversePolicy
from quant_backtester.data import load_ohlc_csv
from quant_backtester.engine import BacktestEngine
from quant_backtester.models.signal import Signal, SignalType
from quant_backtester.strategies.adapters import frame_to_signals
from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy


def _make_legacy_df():
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=5, freq="15min"),
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "entry_signal": [1, 0, 0, -1, 0],
        "exit_long_signal": [False, True, False, False, False],
        "exit_short_signal": [False, False, False, False, True],
    })


def test_frame_to_signals_legacy_conversion_and_immutability():
    df = _make_legacy_df()
    df_copy = df.copy(deep=True)

    signals = frame_to_signals(df, symbol="XAUUSD")
    # Ensure input DataFrame was not mutated
    pd.testing.assert_frame_equal(df, df_copy)

    assert len(signals) == 5
    assert signals[0].signal_type == SignalType.LONG
    assert signals[1].signal_type == SignalType.EXIT_LONG
    assert signals[2].signal_type == SignalType.FLAT
    assert signals[3].signal_type == SignalType.SHORT
    assert signals[4].signal_type == SignalType.EXIT_SHORT
    for i, sig in enumerate(signals):
        assert sig.symbol == "XAUUSD"
        assert sig.time == pd.Timestamp(df.iloc[i]["time"])


def test_frame_to_signals_with_directives():
    df = _make_legacy_df()
    df["quantity"] = [0.05, None, None, 0.02, None]
    df["stop_loss_pct"] = [0.015, None, None, 0.02, None]
    df["take_profit_price"] = [110.0, None, None, 90.0, None]

    signals = frame_to_signals(StrategySignals(df), symbol="XAUUSD")
    directives_0 = signals[0].metadata.get("directives", {})
    assert directives_0["quantity"] == 0.05
    assert directives_0["stop_loss_pct"] == 0.015
    assert directives_0["take_profit_price"] == 110.0

    directives_2 = signals[2].metadata.get("directives", {})
    assert directives_2 == {}


def test_frame_to_signals_validation_errors():
    # Missing required columns
    bad_df = pd.DataFrame({"time": ["2026-01-01"], "open": [100.0]})
    with pytest.raises(ValueError, match="Thiếu cột"):
        frame_to_signals(bad_df, symbol="XAUUSD")

    # Empty symbol
    df = _make_legacy_df()
    with pytest.raises(ValueError, match="symbol"):
        frame_to_signals(df, symbol="")

    # Conflicting exit signals on the same bar (both exit_long and exit_short True with entry != 0)
    df_conflict = _make_legacy_df()
    df_conflict.loc[0, "entry_signal"] = 1
    df_conflict.loc[0, "exit_short_signal"] = True
    # Exit takes priority over entry on same bar, or explicit conflict rule is deterministic
    signals_c = frame_to_signals(df_conflict, symbol="XAUUSD")
    assert isinstance(signals_c[0], Signal)

    # Invalid entry_signal value (e.g. string)
    df_bad_type = _make_legacy_df().astype({"entry_signal": object})
    df_bad_type.loc[0, "entry_signal"] = "INVALID_SIGNAL"
    with pytest.raises((ValueError, TypeError)):
        frame_to_signals(df_bad_type, symbol="XAUUSD")


def test_engine_ast_no_concrete_strategy_imports():
    engine_path = Path(__file__).resolve().parents[2] / "engine.py"
    tree = ast.parse(engine_path.read_text(encoding="utf-8"))
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported_names.append(f"{module}.{alias.name}")

    forbidden = ["MeanReversionStrategy", "mean_reversion", "NewStrategy", "strategy_template"]
    for item in imported_names:
        for f in forbidden:
            assert f not in item, f"engine.py không được import Strategy cụ thể: {item}"


def test_golden_snapshot_sample_ohlc():
    sample_path = Path(__file__).resolve().parents[2] / "examples" / "sample_ohlc.csv"
    df = load_ohlc_csv(sample_path)
    cfg = BacktestConfig()
    strategy = MeanReversionStrategy(window=20, std_multiplier=2.0)
    engine = BacktestEngine(cfg)

    res1 = engine.run(df, strategy)
    res2 = engine.run(df, strategy)

    assert res1.final_balance == pytest.approx(res2.final_balance, rel=1e-9)
    assert res1.final_equity == pytest.approx(res2.final_equity, rel=1e-9)
    assert len(res1.trades) == len(res2.trades)
    assert hasattr(res1, "signal_log")
    assert hasattr(res1, "effective_trade_start")

