import json
from typing import ClassVar
import pandas as pd
import pytest

from quant_backtester.strategies.base import Strategy, StrategySignals
from quant_backtester.strategies.mean_reversion import MeanReversionStrategy
from quant_backtester.strategies.parameters import Parameter, ParameterError, ParameterSchema
from quant_backtester.strategies.registry import StrategyRegistry


class SchemaTestStrategy(Strategy):
    name = "schema_test_strategy"
    parameters: ClassVar[tuple[Parameter, ...]] = (
        Parameter(name="window", param_type=int, default=20, min_value=2, max_value=200, step=1, optimizable=True),
        Parameter(name="threshold", param_type=float, default=1.5, min_value=0.1, max_value=10.0),
        Parameter(name="mode_flag", param_type=str, default="fast", choices=("fast", "slow")),
        Parameter(name="use_filter", param_type=bool, default=False),
    )

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        df["entry_signal"] = 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        return StrategySignals(df)


def test_parameter_schema_valid_and_coercion():
    s = SchemaTestStrategy(window=30, threshold=2)  # int 2 allowed for float
    params = s.get_parameters()
    assert params["window"] == 30
    assert params["threshold"] == 2.0
    assert isinstance(params["threshold"], float)
    assert params["mode_flag"] == "fast"
    assert params["use_filter"] is False

    # JSON serializable
    dumped = json.dumps(params)
    assert '"window": 30' in dumped

    # Coerce mode for CLI strings
    schema = ParameterSchema(SchemaTestStrategy.parameters)
    coerced = schema.validate({"window": "25", "threshold": "2.5", "use_filter": "true"}, coerce=True)
    assert coerced["window"] == 25
    assert coerced["threshold"] == 2.5
    assert coerced["use_filter"] is True


def test_parameter_validation_rules_1_to_10():
    # Rule 1: bool rejected for int/float
    with pytest.raises(ParameterError, match="window"):
        SchemaTestStrategy(window=True)
    with pytest.raises(ParameterError, match="threshold"):
        SchemaTestStrategy(threshold=False)

    # Rule 3: str rejected when coerce=False
    with pytest.raises(ParameterError, match="window"):
        SchemaTestStrategy(window="20")

    # Rule 4: NaN and Inf rejected
    with pytest.raises(ParameterError, match="threshold"):
        SchemaTestStrategy(threshold=float("nan"))
    with pytest.raises(ParameterError, match="threshold"):
        SchemaTestStrategy(threshold=float("inf"))

    # Rule 5: min_value / max_value inclusive bounds
    s_min = SchemaTestStrategy(window=2, threshold=0.1)
    assert s_min.window == 2
    with pytest.raises(ParameterError, match="window"):
        SchemaTestStrategy(window=1)
    with pytest.raises(ParameterError, match="window"):
        SchemaTestStrategy(window=201)

    # Rule 6: choices validation
    with pytest.raises(ParameterError, match="mode_flag"):
        SchemaTestStrategy(mode_flag="invalid_choice")

    # Rule 7 & 8: required=True cannot have default; missing required raises ParameterError
    with pytest.raises(ValueError, match="required"):
        Parameter(name="bad_param", param_type=int, required=True, default=10)

    req_schema = ParameterSchema((Parameter(name="req_p", param_type=int, required=True),))
    with pytest.raises(ParameterError, match="req_p"):
        req_schema.validate({})

    # Rule 9: undeclared parameter raises ParameterError
    with pytest.raises(ParameterError, match="unknown_param"):
        SchemaTestStrategy(unknown_param=123)

    # Rule 10: aggregate multiple errors into one ParameterError
    with pytest.raises(ParameterError) as exc_info:
        SchemaTestStrategy(window=0, threshold=-5.0, unknown_x=1)
    err_msg = str(exc_info.value)
    assert "window" in err_msg
    assert "threshold" in err_msg
    assert "unknown_x" in err_msg


def test_parameter_immutability_after_init():
    s = SchemaTestStrategy(window=20, threshold=2.0)
    # Mutating returned dict must not affect strategy parameters
    p = s.get_parameters()
    p["window"] = 999
    assert s.get_parameters()["window"] == 20

    # Direct attribute assignment to declared parameter should raise AttributeError
    with pytest.raises(AttributeError):
        s.window = 50


def test_registry_validation_and_unregister_and_mean_reversion_compat():
    StrategyRegistry.register("schema_test", SchemaTestStrategy)
    try:
        inst = StrategyRegistry.get("schema_test", window=15)
        assert inst.window == 15

        with pytest.raises(ParameterError):
            StrategyRegistry.get("schema_test", window=-5)
    finally:
        StrategyRegistry.unregister("schema_test")

    assert "schema_test" not in StrategyRegistry.list_strategies()

    # MeanReversionStrategy positional & keyword compatibility
    mr1 = MeanReversionStrategy(25, 2.5)
    assert mr1.window == 25
    assert mr1.std_multiplier == 2.5
    assert mr1.get_parameters() == {"window": 25, "std_multiplier": 2.5}

    with pytest.raises((ParameterError, ValueError)):
        MeanReversionStrategy(1, -1.0)

