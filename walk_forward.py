"""Backward-compatible Walk-Forward wrapper ủy quyền sang quant_backtester.validation.walk_forward."""
from __future__ import annotations

import pandas as pd

from .strategies.mean_reversion import MeanReversionStrategy
from .validation.walk_forward import WalkForwardMode, run_walk_forward


def walk_forward_mean_reversion(
    data: pd.DataFrame,
    config,
    windows=(15, 20, 25, 30),
    multipliers=(1.5, 2.0, 2.5),
    folds: int = 5,
):
    if folds < 3:
        raise ValueError("folds phải >= 3.")
    wf = run_walk_forward(
        market_data=data,
        strategy_or_cls=MeanReversionStrategy,
        config=config,
        param_grid={"window": list(windows), "std_multiplier": list(multipliers)},
        folds=folds,
        mode=WalkForwardMode.ROLLING,
        objective_metric="sharpe_annual",
        min_train_trades=0,
    )
    if wf.folds_df.empty:
        return wf.folds_df, pd.DataFrame()
    return wf.folds_df, wf.summary_df
