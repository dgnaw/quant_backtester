"""Extensible educational backtesting framework."""
from .config import BacktestConfig
from .engine import BacktestEngine
from .strategies.mean_reversion import MeanReversionStrategy

__all__ = ["BacktestConfig", "BacktestEngine", "MeanReversionStrategy"]
