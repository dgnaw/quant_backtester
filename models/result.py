from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from quant_backtester.config import EngineConfig


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict[str, Any]
    config: EngineConfig
    strategy_name: str
    strategy_parameters: dict[str, Any] = field(default_factory=dict)
    signal_log: list[dict[str, Any]] = field(default_factory=list)
    effective_trade_start: int = 0

    @property
    def final_balance(self) -> float:
        if not self.equity.empty and "balance" in self.equity.columns:
            return float(self.equity["balance"].iloc[-1])
        return self.config.account.initial_balance

    @property
    def final_equity(self) -> float:
        if not self.equity.empty and "equity" in self.equity.columns:
            return float(self.equity["equity"].iloc[-1])
        return self.config.account.initial_balance

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "trades": self.trades,
            "orders": self.orders,
            "metrics": self.metrics,
            "final_balance": self.final_balance,
            "final_equity": self.final_equity,
            "strategy": self.strategy_name,
            "strategy_parameters": self.strategy_parameters,
            "signal_log": self.signal_log,
            "effective_trade_start": self.effective_trade_start,
        }

    # Backward compatibility: hỗ trợ result["equity"], result["trades"]
    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def __contains__(self, item: str) -> bool:
        return item in self.to_dict()
