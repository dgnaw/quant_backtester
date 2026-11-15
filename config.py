from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

class ReversePolicy(Enum):
    IGNORE = "IGNORE"
    REVERSE = "REVERSE"
    TWO_ORDER_REVERSE = "REVERSE"


class CommissionMode(Enum):
    PER_SIDE = "PER_SIDE"
    ROUND_TURN = "ROUND_TURN"


class IntrabarConflictPolicy(Enum):
    WORST_CASE_SL_FIRST = "WORST_CASE_SL_FIRST"
    BEST_CASE_TP_FIRST = "BEST_CASE_TP_FIRST"


class PositionMode(Enum):
    NETTING_SINGLE = "NETTING_SINGLE"
    NETTING_SCALE_IN = "NETTING_SCALE_IN"


class SizingMode(Enum):
    FIXED_LOT = "FIXED_LOT"
    RISK_BASED = "RISK_BASED"


class SizingBasis(Enum):
    EQUITY = "EQUITY"
    BALANCE = "BALANCE"


class MissingCorrelationPolicy(Enum):
    REJECT = "REJECT"
    IGNORE = "IGNORE"
    CONSERVATIVE_ONE = "CONSERVATIVE_ONE"


@dataclass
class AccountConfig:
    initial_balance: float = 1_000.0
    base_currency: str = "USD"
    account_currency: str | None = None
    leverage: float = 100.0

    def __post_init__(self):
        if self.initial_balance <= 0 or not math.isfinite(float(self.initial_balance)):
            raise ValueError("initial_balance phải lớn hơn 0.")
        if self.leverage <= 0 or not math.isfinite(float(self.leverage)):
            raise ValueError("leverage phải lớn hơn 0 và hữu hạn.")
        if self.account_currency is not None:
            self.base_currency = str(self.account_currency)
        else:
            self.account_currency = self.base_currency


@dataclass
class InstrumentConfig:
    symbol: str = "XAUUSD"
    base_currency: str = "XAU"
    lot_size: float = 0.01
    contract_size: float = 100.0
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    point_size: float = 0.01
    quote_currency: str = "USD"
    margin_currency: str | None = None
    leverage: float | None = None

    def __post_init__(self):
        if self.lot_size <= 0 or self.contract_size <= 0:
            raise ValueError("lot_size và contract_size phải lớn hơn 0.")
        if self.min_lot <= 0 or self.max_lot < self.min_lot or self.lot_step <= 0:
            raise ValueError("min_lot, max_lot, lot_step không hợp lệ.")
        if self.point_size <= 0:
            raise ValueError("point_size phải lớn hơn 0.")
        if self.leverage is not None and (
            not math.isfinite(float(self.leverage)) or float(self.leverage) <= 0
        ):
            raise ValueError("instrument.leverage phải lớn hơn 0 và hữu hạn.")

    @property
    def effective_margin_currency(self) -> str:
        return self.margin_currency if self.margin_currency is not None else self.quote_currency

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
class FeeConfig:
    fee_rate_per_side: float = 0.00015
    commission_per_lot: float = 0.0
    commission_mode: str | CommissionMode = "ROUND_TURN"

    def __post_init__(self):
        if self.fee_rate_per_side < 0 or self.commission_per_lot < 0:
            raise ValueError("fee_rate_per_side và commission_per_lot không được âm.")
        if isinstance(self.commission_mode, CommissionMode):
            self.commission_mode = self.commission_mode.value


@dataclass
class SlippageConfig:
    slippage_price: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self):
        if self.slippage_price < 0 or self.slippage_rate < 0:
            raise ValueError("slippage_price và slippage_rate không được âm.")


@dataclass
class RiskConfig:
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02

    # F1: Position Sizing
    default_sizing_mode: SizingMode = SizingMode.FIXED_LOT
    sizing_mode: SizingMode | None = None
    sizing_basis: SizingBasis = SizingBasis.EQUITY
    risk_per_trade_pct: float | None = None
    risk_per_trade_amount: float | None = None
    clamp_to_max_lot: bool = False
    include_fees_in_risk: bool = True
    include_slippage_in_risk: bool = False

    # F2: Per-Trade Risk Cap
    max_risk_per_trade_pct: float | None = None
    max_risk_per_trade_amount: float | None = None

    # F3: Daily Loss Limit
    max_daily_loss_pct: float | None = None
    max_daily_loss_amount: float | None = None
    trading_timezone: str = "UTC"

    # F4: Maximum Drawdown Circuit Breaker
    max_drawdown_pct: float | None = None
    max_drawdown_amount: float | None = None
    liquidate_on_max_drawdown: bool = False

    # F5: Symbol & Account Exposure Limits
    max_symbol_lots: float | None = None
    max_account_lots: float | None = None
    max_symbol_notional: float | None = None
    max_account_notional: float | None = None
    include_pending_orders_in_exposure: bool = True

    # F6: Margin, Leverage & Stop-Out
    leverage: float | None = None
    enable_margin_check: bool = False
    enforce_margin_check: bool = False
    reserve_margin_for_pending_orders: bool = True
    margin_call_level_pct: float | None = None
    stop_out_level_pct: float | None = None

    # F7: Correlation Risk
    enable_correlation_check: bool = False
    correlation_matrix: dict[Any, Any] = field(default_factory=dict)
    correlation_threshold: float = 0.70
    max_correlated_positions: int | None = None
    max_correlated_notional: float | None = None
    missing_correlation_policy: MissingCorrelationPolicy = MissingCorrelationPolicy.REJECT

    def __post_init__(self):
        if self.stop_loss_pct < 0 or self.take_profit_pct < 0:
            raise ValueError("stop_loss_pct và take_profit_pct không được âm.")
        if self.sizing_mode is not None:
            self.default_sizing_mode = (
                self.sizing_mode
                if isinstance(self.sizing_mode, SizingMode)
                else SizingMode(self.sizing_mode)
            )
        if not isinstance(self.default_sizing_mode, SizingMode):
            self.default_sizing_mode = SizingMode(self.default_sizing_mode)
        self.sizing_mode = self.default_sizing_mode

        if not isinstance(self.sizing_basis, SizingBasis):
            self.sizing_basis = SizingBasis(self.sizing_basis)
        if not isinstance(self.missing_correlation_policy, MissingCorrelationPolicy):
            self.missing_correlation_policy = MissingCorrelationPolicy(
                self.missing_correlation_policy
            )

        if self.enable_margin_check or self.enforce_margin_check:
            self.enable_margin_check = True
            self.enforce_margin_check = True

        if self.max_correlated_positions is not None or self.max_correlated_notional is not None:
            self.enable_correlation_check = True

        if self.risk_per_trade_pct is not None and self.risk_per_trade_amount is not None:
            raise ValueError(
                "CONFLICTING_RISK_BUDGET_CONFIG: Không được cấu hình đồng thời cả risk_per_trade_pct và risk_per_trade_amount."
            )

        positive_optional_fields = (
            ("risk_per_trade_pct", self.risk_per_trade_pct),
            ("risk_per_trade_amount", self.risk_per_trade_amount),
            ("max_risk_per_trade_pct", self.max_risk_per_trade_pct),
            ("max_risk_per_trade_amount", self.max_risk_per_trade_amount),
            ("max_daily_loss_pct", self.max_daily_loss_pct),
            ("max_daily_loss_amount", self.max_daily_loss_amount),
            ("max_drawdown_pct", self.max_drawdown_pct),
            ("max_drawdown_amount", self.max_drawdown_amount),
            ("max_symbol_lots", self.max_symbol_lots),
            ("max_account_lots", self.max_account_lots),
            ("max_symbol_notional", self.max_symbol_notional),
            ("max_account_notional", self.max_account_notional),
            ("leverage", self.leverage),
            ("margin_call_level_pct", self.margin_call_level_pct),
            ("stop_out_level_pct", self.stop_out_level_pct),
            ("max_correlated_notional", self.max_correlated_notional),
        )
        for fname, val in positive_optional_fields:
            if val is not None:
                if (
                    isinstance(val, bool)
                    or not isinstance(val, (int, float))
                    or not math.isfinite(float(val))
                    or float(val) <= 0.0
                ):
                    raise ValueError(
                        f"RiskConfig.{fname} phải lớn hơn 0 và hữu hạn (nhận {val!r})."
                    )

        if self.max_correlated_positions is not None:
            if (
                isinstance(self.max_correlated_positions, bool)
                or not isinstance(self.max_correlated_positions, int)
                or self.max_correlated_positions <= 0
            ):
                raise ValueError(
                    "RiskConfig.max_correlated_positions phải là số nguyên dương (> 0)."
                )

        if not math.isfinite(float(self.correlation_threshold)) or not (
            0.0 < float(self.correlation_threshold) <= 1.0
        ):
            raise ValueError(
                "RiskConfig.correlation_threshold phải nằm trong khoảng (0.0, 1.0]."
            )

        if (
            self.margin_call_level_pct is not None
            and self.stop_out_level_pct is not None
            and float(self.stop_out_level_pct) > float(self.margin_call_level_pct) + 1e-9
        ):
            raise ValueError("stop_out_level_pct không được lớn hơn margin_call_level_pct.")

        # Chuẩn hóa correlation_matrix: hỗ trợ cả dict[(sym_a, sym_b), float] và dict[sym_a, dict[sym_b, float]]
        norm_corr: dict[tuple[str, str], float] = {}
        raw_matrix = self.correlation_matrix or {}
        flat_items: list[tuple[str, str, Any]] = []
        for k, v in raw_matrix.items():
            if isinstance(k, tuple) and len(k) == 2:
                flat_items.append((str(k[0]), str(k[1]), v))
            elif isinstance(k, str) and isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_items.append((str(k), str(sub_k), sub_v))
            else:
                raise ValueError(
                    f"Khóa của correlation_matrix phải là tuple (sym_a, sym_b) hoặc nested dict, nhận {k!r}."
                )

        for sym_a, sym_b, rho in flat_items:
            if (
                isinstance(rho, bool)
                or not isinstance(rho, (int, float))
                or not math.isfinite(float(rho))
            ):
                raise ValueError(
                    f"Hệ số tương quan cho ({sym_a}, {sym_b}) phải là số thực hữu hạn."
                )
            r_val = float(rho)
            if r_val < -1.0 - 1e-9 or r_val > 1.0 + 1e-9:
                raise ValueError(
                    f"Hệ số tương quan cho ({sym_a}, {sym_b}) phải nằm trong [-1.0, 1.0] (nhận {r_val})."
                )
            r_val = max(-1.0, min(1.0, r_val))
            if sym_a == sym_b and not math.isclose(r_val, 1.0, abs_tol=1e-9):
                raise ValueError(
                    f"Tương quan của một symbol với chính nó ({sym_a}, {sym_b}) phải bằng 1.0."
                )
            if (sym_b, sym_a) in norm_corr and not math.isclose(
                norm_corr[(sym_b, sym_a)], r_val, abs_tol=1e-9
            ):
                raise ValueError(
                    f"ASYMMETRIC_CORRELATION_MATRIX: Tương quan ({sym_a}, {sym_b})={r_val} mâu thuẫn với ({sym_b}, {sym_a})={norm_corr[(sym_b, sym_a)]}."
                )
            norm_corr[(sym_a, sym_b)] = r_val
            norm_corr[(sym_b, sym_a)] = r_val
        self.correlation_matrix = norm_corr


@dataclass
class AnalyticsConfig:
    risk_free_rate_annual: float = 0.0
    timeframe: str | None = None
    auto_annualize_from_data: bool = False
    trading_days_per_year: int = 252
    trading_hours_per_day: float = 24.0
    enable_buy_and_hold_benchmark: bool = True

    def __post_init__(self):
        if (
            isinstance(self.risk_free_rate_annual, bool)
            or not isinstance(self.risk_free_rate_annual, (int, float))
            or not math.isfinite(float(self.risk_free_rate_annual))
            or float(self.risk_free_rate_annual) <= -1.0
        ):
            raise ValueError("risk_free_rate_annual phải là số thực hữu hạn và > -1.0.")
        self.risk_free_rate_annual = float(self.risk_free_rate_annual)
        if (
            isinstance(self.trading_days_per_year, bool)
            or not isinstance(self.trading_days_per_year, int)
            or self.trading_days_per_year <= 0
        ):
            raise ValueError("trading_days_per_year phải là số nguyên dương (> 0).")
        if (
            isinstance(self.trading_hours_per_day, bool)
            or not isinstance(self.trading_hours_per_day, (int, float))
            or not math.isfinite(float(self.trading_hours_per_day))
            or float(self.trading_hours_per_day) <= 0.0
            or float(self.trading_hours_per_day) > 24.0
        ):
            raise ValueError("trading_hours_per_day phải nằm trong khoảng (0.0, 24.0].")
        self.trading_hours_per_day = float(self.trading_hours_per_day)
        if self.timeframe is not None:
            self.timeframe = str(self.timeframe).strip().upper()
            if not self.timeframe:
                self.timeframe = None


@dataclass
class EngineConfig:
    account: AccountConfig = field(default_factory=AccountConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    fee: FeeConfig = field(default_factory=FeeConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    bars_per_year: int = 96 * 252
    risk_free_rate_annual: float | None = None
    timeframe: str | None = None
    auto_annualize_from_data: bool | None = None
    force_close_at_end: bool = True
    reverse_policy: ReversePolicy = ReversePolicy.IGNORE
    intrabar_conflict_policy: IntrabarConflictPolicy = IntrabarConflictPolicy.WORST_CASE_SL_FIRST
    limit_penetration: float = 0.0
    default_position_mode: PositionMode = PositionMode.NETTING_SINGLE
    position_mode_by_symbol: dict[str, PositionMode] = field(default_factory=dict)

    def __post_init__(self):
        if self.bars_per_year <= 0:
            raise ValueError("bars_per_year phải lớn hơn 0.")
        if self.limit_penetration < 0:
            raise ValueError("limit_penetration không được âm.")
        if not isinstance(self.default_position_mode, PositionMode):
            self.default_position_mode = PositionMode(self.default_position_mode)
        normalized_map: dict[str, PositionMode] = {}
        for sym, mode in (self.position_mode_by_symbol or {}).items():
            normalized_map[str(sym)] = (
                mode if isinstance(mode, PositionMode) else PositionMode(mode)
            )
        self.position_mode_by_symbol = normalized_map
        if self.risk_free_rate_annual is not None:
            if (
                isinstance(self.risk_free_rate_annual, bool)
                or not isinstance(self.risk_free_rate_annual, (int, float))
                or not math.isfinite(float(self.risk_free_rate_annual))
                or float(self.risk_free_rate_annual) <= -1.0
            ):
                raise ValueError("risk_free_rate_annual phải là số thực hữu hạn và > -1.0.")
            self.analytics.risk_free_rate_annual = float(self.risk_free_rate_annual)
        else:
            self.risk_free_rate_annual = self.analytics.risk_free_rate_annual
        if self.timeframe is not None:
            self.analytics.timeframe = str(self.timeframe).strip().upper()
            self.timeframe = self.analytics.timeframe
        else:
            self.timeframe = self.analytics.timeframe
        if self.auto_annualize_from_data is not None:
            self.analytics.auto_annualize_from_data = bool(self.auto_annualize_from_data)
        else:
            self.auto_annualize_from_data = self.analytics.auto_annualize_from_data

    def get_position_mode(self, symbol: str) -> PositionMode:
        return self.position_mode_by_symbol.get(symbol, self.default_position_mode)

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
        default_position_mode: PositionMode = PositionMode.NETTING_SINGLE,
        position_mode_by_symbol: dict[str, PositionMode] | None = None,
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
            fee=FeeConfig(
                fee_rate_per_side=fee_rate_per_side, commission_per_lot=commission_per_lot
            ),
            slippage=SlippageConfig(slippage_price=slippage_price),
            risk=RiskConfig(stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct),
            bars_per_year=bars_per_year,
            force_close_at_end=force_close_at_end,
            reverse_policy=reverse_policy,
            intrabar_conflict_policy=intrabar_conflict_policy,
            limit_penetration=limit_penetration,
            default_position_mode=default_position_mode,
            position_mode_by_symbol=position_mode_by_symbol or {},
        )

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
