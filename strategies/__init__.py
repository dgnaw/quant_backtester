from .base import BaseStrategy, Strategy, StrategyMode, StrategySignals
from .context import FillView, StrategyContext
from .parameters import Parameter, ParameterError, ParameterSchema
from .adapters import frame_to_signals
from .translator import SignalTranslator
from .registry import StrategyRegistry
from .mean_reversion import MeanReversionStrategy

# Tự động đăng ký MeanReversionStrategy vào Registry
StrategyRegistry.register("mean_reversion", MeanReversionStrategy)

__all__ = [
    "BaseStrategy",
    "Strategy",
    "StrategyMode",
    "StrategySignals",
    "StrategyContext",
    "FillView",
    "Parameter",
    "ParameterSchema",
    "ParameterError",
    "frame_to_signals",
    "SignalTranslator",
    "StrategyRegistry",
    "MeanReversionStrategy",
]
