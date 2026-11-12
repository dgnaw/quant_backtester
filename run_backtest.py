import argparse
from pathlib import Path
import pandas as pd
from .config import BacktestConfig
from .data import load_ohlc_csv
from .engine import BacktestEngine
from .analytics import calculate_metrics, print_report
from .strategies.mean_reversion import MeanReversionStrategy
from .walk_forward import walk_forward_mean_reversion


def main():
    parser = argparse.ArgumentParser(description="Extensible Quant Backtesting Framework")
    parser.add_argument("csv", help="CSV with time,open,high,low,close")
    parser.add_argument("--strategy", choices=["mean_reversion"], default="mean_reversion")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--std-multiplier", type=float, default=2.0)
    parser.add_argument("--initial-balance", type=float, default=1000)
    parser.add_argument("--lot-size", type=float, default=0.01)
    parser.add_argument("--contract-size", type=float, default=100)
    parser.add_argument("--fee-rate", type=float, default=0.00015)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    data = load_ohlc_csv(args.csv)
    config = BacktestConfig(initial_balance=args.initial_balance, lot_size=args.lot_size,
                            contract_size=args.contract_size, fee_rate_per_side=args.fee_rate,
                            slippage_price=args.slippage)
    strategy = MeanReversionStrategy(args.window, args.std_multiplier)
    # Chronological split. Warm-up bars calculate indicators but cannot trade.
    split = int(len(data) * 0.7)
    train = data.iloc[:split].reset_index(drop=True)
    wp = strategy.warmup_period
    test = data.iloc[max(0, split - wp):].reset_index(drop=True)
    engine = BacktestEngine(config)
    train_result = engine.run(train, strategy)
    test_result = engine.run(test, strategy, trade_start=min(wp, len(test) - 1))
    print_report(calculate_metrics(train_result, config), "IN-SAMPLE / TRAIN")
    print_report(calculate_metrics(test_result, config), "OUT-OF-SAMPLE / TEST")
    Path("output").mkdir(exist_ok=True)
    train_result["trades"].to_csv("output/trades_train.csv", index=False)
    test_result["trades"].to_csv("output/trades_test.csv", index=False)
    train_result["equity"].to_csv("output/equity_train.csv", index=False)
    test_result["equity"].to_csv("output/equity_test.csv", index=False)
    if args.walk_forward:
        folds, summary = walk_forward_mean_reversion(data, config)
        folds.to_csv("output/walk_forward_folds.csv", index=False)
        summary.to_csv("output/walk_forward_summary.csv", index=False)
        print("\nWALK-FORWARD FOLDS\n", folds.to_string(index=False))
        print("\nWALK-FORWARD SUMMARY\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
