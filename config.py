from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ReversePolicy(Enum):
    IGNORE = "IGNORE"
    REVERSE = "REVERSE"


class IntrabarConflictPolicy(Enum):
    WORST_CASE_SL_FIRST = "WORST_CASE_SL_FIRST"
    BEST_CASE_TP_FIRST = "BEST_CASE_TP_FIRST"


@dataclass
# Cấu hình vốn ban đầu, đồng tiền cơ sở
class AccountConfig:
    initial_balance: float = 1_000.0
    base_currency: str = "USD"

    def __post_init__(self):
        if self.initial_balance <= 0:
            raise ValueError("initial_balance phải lớn hơn 0.")


@dataclass
class InstrumentConfig:
    symbol: str = "XAUUSD"
    lot_size: float = 0.01
    contract_size: float = 100.0
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    point_size: float = 0.01
    quote_currency: str = "USD"

    def __post_init__(self):
        if self.lot_size <= 0 or self.contract_size <= 0:
            raise ValueError("lot_size và contract_size phải lớn hơn 0.")
        if self.min_lot <= 0 or self.max_lot < self.min_lot or self.lot_step <= 0:
            raise ValueError("min_lot, max_lot, lot_step không hợp lệ.")
        if self.point_size <= 0:
            raise ValueError("point_size phải lớn hơn 0.")

    def lots_to_units(self, quantity: float) -> float:
        return float(quantity) * self.contract_size

    @property
    def units(self) -> float:
        return self.lots_to_units(self.lot_size)


@dataclass
class SpreadConfig:
    spread_price: float = 0.0
    spread_points: float | None = None
    price_type: str = "MID"

    def __post_init__(self):
        if self.spread_price < 0:
            raise ValueError("spread_price không được âm.")
        if self.spread_points is not None and self.spread_points < 0:
            raise ValueError("spread_points không được âm.")


@dataclass
# Tỷ lệ phí hoa hồng
class FeeConfig:
    fee_rate_per_side: float = 0.00015
    commission_per_lot: float = 0.0
    commission_mode: str = "ROUND_TURN"

    def __post_init__(self):
        if self.fee_rate_per_side < 0 or self.commission_per_lot < 0:
            raise ValueError("fee_rate_per_side và commission_per_lot không được âm.")


@dataclass
# Độ trượt giá thị trường
class SlippageConfig:
    slippage_price: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self):
        if self.slippage_price < 0 or self.slippage_rate < 0:
            raise ValueError("slippage_price và slippage_rate không được âm.")


@dataclass
# Tỷ lệ cắt lỗ/chốt lời
class RiskConfig:
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02

    def __post_init__(self):
        if self.stop_loss_pct < 0 or self.take_profit_pct < 0:
            raise ValueError("stop_loss_pct và take_profit_pct không được âm.")


@dataclass
class EngineConfig:
    account: AccountConfig = field(default_factory=AccountConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    fee: FeeConfig = field(default_factory=FeeConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    bars_per_year: int = 96 * 252  # M15 assumption
    force_close_at_end: bool = True
    reverse_policy: ReversePolicy = ReversePolicy.IGNORE
    intrabar_conflict_policy: IntrabarConflictPolicy = IntrabarConflictPolicy.WORST_CASE_SL_FIRST
    limit_penetration: float = 0.0

    def __post_init__(self):
        if self.bars_per_year <= 0:
            raise ValueError("bars_per_year phải lớn hơn 0.")
        if self.limit_penetration < 0:
            raise ValueError("limit_penetration không được âm.")

    @property
    def units(self) -> float:
        return self.instrument.units

    @property
    def initial_balance(self) -> float:
        return self.account.initial_balance


class BacktestConfig(EngineConfig):
    """Lớp kế thừa để đảm bảo backward compatibility với code cũ."""
    def __init__(
        self,
        initial_balance: float = 1_000.0,
        lot_size: float = 0.01,
        contract_size: float = 100.0,
        fee_rate_per_side: float = 0.00015,
        slippage_price: float = 0.0,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
        bars_per_year: int = 96 * 252,
        force_close_at_end: bool = True,
        symbol: str = "XAUUSD",
        base_currency: str = "USD",
        commission_per_lot: float = 0.0,
        spread_price: float = 0.0,
        point_size: float = 0.01,
        reverse_policy: ReversePolicy = ReversePolicy.IGNORE,
        intrabar_conflict_policy: IntrabarConflictPolicy = IntrabarConflictPolicy.WORST_CASE_SL_FIRST,
        limit_penetration: float = 0.0,
    ):
        super().__init__(
            account=AccountConfig(initial_balance=initial_balance, base_currency=base_currency),
            instrument=InstrumentConfig(
                symbol=symbol,
                lot_size=lot_size,
                contract_size=contract_size,
                point_size=point_size,
            ),
            spread=SpreadConfig(spread_price=spread_price),
            fee=FeeConfig(fee_rate_per_side=fee_rate_per_side, commission_per_lot=commission_per_lot),
            slippage=SlippageConfig(slippage_price=slippage_price),
            risk=RiskConfig(stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct),
            bars_per_year=bars_per_year,
            force_close_at_end=force_close_at_end,
            reverse_policy=reverse_policy,
            intrabar_conflict_policy=intrabar_conflict_policy,
            limit_penetration=limit_penetration,
        )

    # Property alias để code cũ truy cập trực tiếp
    @property
    def initial_balance(self) -> float:
        return self.account.initial_balance

    @property
    def lot_size(self) -> float:
        return self.instrument.lot_size

    @property
    def contract_size(self) -> float:
        return self.instrument.contract_size

    @property
    def fee_rate_per_side(self) -> float:
        return self.fee.fee_rate_per_side

    @property
    def slippage_price(self) -> float:
        return self.slippage.slippage_price

    @property
    def stop_loss_pct(self) -> float:
        return self.risk.stop_loss_pct

    @property
    def take_profit_pct(self) -> float:
        return self.risk.take_profit_pct
