"""Generate deterministic sample OHLC data for trying the framework."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    n = 3000
    time = pd.date_range("2025-01-01", periods=n, freq="15min")
    close = 2650 + np.cumsum(rng.normal(0, 1.8, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.2, n)
    out = pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close})
    target = Path(__file__).with_name("sample_ohlc.csv")
    out.to_csv(target, index=False)
    print(f"Created: {target}")


if __name__ == "__main__":
    main()
