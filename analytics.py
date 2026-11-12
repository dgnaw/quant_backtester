import math
import numpy as np


def calculate_metrics(result: dict, config) -> dict:
    equity = result["equity"]["equity"].astype(float)
    if equity.empty:
        raise ValueError("Equity curve rỗng.")
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    std = returns.std(ddof=1)
    sharpe = returns.mean() / std * math.sqrt(config.bars_per_year) if pd_valid(std) and std > 0 else 0.0
    trades = result["trades"]
    if trades.empty:
        win_rate = profit_factor = expectancy = 0.0
        wins = losses = 0
    else:
        pnl = trades["net_pnl"].astype(float)
        wins, losses = int((pnl > 0).sum()), int((pnl < 0).sum())
        win_rate = wins / len(pnl) * 100
        gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
        profit_factor = gp / gl if gl else (float("inf") if gp else 0.0)
        expectancy = float(pnl.mean())
    return {
        "initial_balance": config.initial_balance,
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": (float(equity.iloc[-1]) / config.initial_balance - 1) * 100,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "sharpe_annual": float(sharpe),
        "total_trades": int(len(trades)), "wins": wins, "losses": losses,
        "win_rate_pct": float(win_rate), "profit_factor": float(profit_factor),
        "expectancy_cash_per_trade": float(expectancy),
    }


def pd_valid(value) -> bool:
    return value is not None and np.isfinite(value)


def print_report(metrics: dict, title: str = "BACKTEST REPORT"):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)
    for key, label in [
        ("initial_balance", "Initial balance"), ("final_equity", "Final equity"),
        ("total_return_pct", "Total return (%)"), ("max_drawdown_pct", "Max drawdown (%)"),
        ("sharpe_annual", "Annualized Sharpe"), ("total_trades", "Total trades"),
        ("win_rate_pct", "Win rate (%)"), ("profit_factor", "Profit factor"),
        ("expectancy_cash_per_trade", "Expectancy / trade"),
    ]:
        print(f"{label:<25}: {metrics[key]:,.4f}")
