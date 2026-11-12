"""Simple rolling walk-forward parameter selection for strategies exposing parameter_grid()."""
from itertools import product
import numpy as np
import pandas as pd
from .engine import BacktestEngine
from .analytics import calculate_metrics
from .strategies.mean_reversion import MeanReversionStrategy


def walk_forward_mean_reversion(data: pd.DataFrame, config, windows=(15, 20, 25, 30),
                                multipliers=(1.5, 2.0, 2.5), folds: int = 5):
    if folds < 3:
        raise ValueError("folds phải >= 3.")
    edges = np.linspace(0, len(data), folds + 1, dtype=int)
    fold_rows, oos_series = [], []
    for fold in range(folds - 1):
        train = data.iloc[edges[fold]:edges[fold + 1]].reset_index(drop=True)
        test = data.iloc[edges[fold + 1]:edges[fold + 2]].reset_index(drop=True)
        if len(train) <= max(windows) + 5 or test.empty:
            continue
        best = None
        for window, mult in product(windows, multipliers):
            strategy = MeanReversionStrategy(window=window, std_multiplier=mult)
            result = BacktestEngine(config).run(train, strategy)
            metrics = calculate_metrics(result, config)
            score = metrics["sharpe_annual"]
            if best is None or score > best[0]:
                best = (score, strategy)
        if best is None:
            continue
        selected = best[1]
        wp = selected.warmup_period
        warmup = data.iloc[max(0, edges[fold + 1] - wp):edges[fold + 1]]
        combined = pd.concat([warmup, test], ignore_index=True)
        result = BacktestEngine(config).run(combined, selected, trade_start=len(warmup))
        metrics = calculate_metrics(result, config)
        returns = result["equity"]["equity"].pct_change().fillna(0.0)
        oos_series.append(returns.reset_index(drop=True))
        fold_rows.append({
            "fold": fold + 1, "window": selected.window, "std_multiplier": selected.std_multiplier,
            "train_sharpe": best[0], "test_return_pct": metrics["total_return_pct"],
            "test_max_drawdown_pct": metrics["max_drawdown_pct"], "test_trades": metrics["total_trades"],
        })
    folds_df = pd.DataFrame(fold_rows)
    if not oos_series:
        return folds_df, pd.DataFrame()
    all_returns = pd.concat(oos_series, ignore_index=True)
    oos_equity = (1 + all_returns).cumprod()
    peak = oos_equity.cummax()
    std = all_returns.std(ddof=1)
    sharpe = all_returns.mean() / std * np.sqrt(config.bars_per_year) if pd.notna(std) and std > 0 else 0.0
    summary = pd.DataFrame([{
        "combined_oos_return_pct": (oos_equity.iloc[-1] - 1) * 100,
        "combined_oos_max_drawdown_pct": ((oos_equity / peak) - 1).min() * 100,
        "combined_oos_sharpe": sharpe, "oos_bars": len(all_returns),
    }])
    return folds_df, summary
