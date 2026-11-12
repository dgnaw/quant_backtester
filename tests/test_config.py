import pytest
from dataclasses import asdict
from quant_backtester.config import (
    AccountConfig,
    InstrumentConfig,
    FeeConfig,
    SlippageConfig,
    RiskConfig,
    EngineConfig,
    BacktestConfig,  # backward compatibility alias
)


def test_account_config_validation():
    cfg = AccountConfig(initial_balance=1000.0, base_currency="USD")
    assert cfg.initial_balance == 1000.0
    assert cfg.base_currency == "USD"

    with pytest.raises(ValueError, match="initial_balance phải lớn hơn 0"):
        AccountConfig(initial_balance=0.0)


def test_instrument_config_validation():
    cfg = InstrumentConfig(symbol="XAUUSD", lot_size=0.01, contract_size=100.0)
    assert cfg.units == pytest.approx(1.0)

    with pytest.raises(ValueError, match="lot_size và contract_size phải lớn hơn 0"):
        InstrumentConfig(symbol="XAUUSD", lot_size=-0.01)


def test_fee_and_slippage_config_validation():
    fee_cfg = FeeConfig(fee_rate_per_side=0.00015)
    assert fee_cfg.fee_rate_per_side == 0.00015

    with pytest.raises(ValueError, match="fee_rate_per_side"):
        FeeConfig(fee_rate_per_side=-0.0001)

    slip_cfg = SlippageConfig(slippage_price=0.05)
    assert slip_cfg.slippage_price == 0.05

    with pytest.raises(ValueError, match="slippage_price"):
        SlippageConfig(slippage_price=-0.01)


def test_risk_config_validation():
    risk_cfg = RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02)
    assert risk_cfg.stop_loss_pct == 0.01
    assert risk_cfg.take_profit_pct == 0.02

    with pytest.raises(ValueError, match="stop_loss_pct và take_profit_pct"):
        RiskConfig(stop_loss_pct=-0.01)


def test_engine_config_composition():
    cfg = EngineConfig()
    assert cfg.account.initial_balance == 1000.0
    assert cfg.instrument.symbol == "XAUUSD"
    assert cfg.fee.fee_rate_per_side == 0.00015
    assert cfg.slippage.slippage_price == 0.0
    assert cfg.risk.stop_loss_pct == 0.01
    assert cfg.units == pytest.approx(1.0)


def test_backward_compatibility_backtest_config():
    # Code cũ khởi tạo BacktestConfig(initial_balance=1000, lot_size=0.01, contract_size=100.0) vẫn phải hoạt động
    cfg = BacktestConfig(initial_balance=5000.0, lot_size=0.1, fee_rate_per_side=0.0002)
    assert cfg.initial_balance == 5000.0
    assert cfg.lot_size == 0.1
    assert cfg.fee_rate_per_side == 0.0002
    assert cfg.account.initial_balance == 5000.0
    assert cfg.instrument.lot_size == 0.1
    assert cfg.units == pytest.approx(10.0)

