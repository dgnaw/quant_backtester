"""H6 – Execution Friction Stress Testing (Spread, Commission, Slippage & Break-Even Analysis)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_backtester.analytics.accounting import summarize_ledger_cash_flows
from quant_backtester.config import EngineConfig
from quant_backtester.engine import BacktestEngine
from quant_backtester.execution.fee import BaseFeeModel
from quant_backtester.models.order import OrderSide
from quant_backtester.validation.grid_runner import (
    clone_config,
    instantiate_fresh_strategy,
    run_backtest,
)
from quant_backtester.validation.models import ValidationStatus


@dataclass(frozen=True)
class StressScenarioResult:
    scenario_name: str
    mode: str          # CONTROLLED_COMMISSION | FULL_ENGINE
    stress_type: str   # SPREAD | COMMISSION | SLIPPAGE | COMBINED
    multiplier: float
    spread_price: float
    spread_points: float | None
    fee_rate_per_side: float
    commission_per_lot: float
    slippage_price: float
    slippage_rate: float
    total_return_pct: float
    net_trading_profit: float
    max_drawdown_pct: float
    sharpe_annual: float
    profit_factor: float
    total_trades: int
    total_fees: float
    total_slippage_cost: float


@dataclass
class FrictionStressReport:
    status: ValidationStatus
    scenarios_df: pd.DataFrame
    controlled_commission_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    break_even_multipliers: dict[str, float | None] = field(default_factory=dict)
    break_even_multiplier: float | None = None
    is_controlled_commission_monotonic: bool = True
    messages: list[str] = field(default_factory=list)


def _scale_config(config: EngineConfig, stress_type: str, mult: float) -> EngineConfig:
    st = str(stress_type).strip().upper()
    if st not in {"SPREAD", "COMMISSION", "SLIPPAGE", "COMBINED"}:
        raise ValueError(f"stress_type '{stress_type}' không được hỗ trợ.")
    m = float(mult)
    if not math.isfinite(m) or m < 0.0:
        raise ValueError(f"multiplier phải >= 0 và hữu hạn (nhận {mult!r}).")

    cfg = clone_config(config)
    if st in {"SPREAD", "COMBINED"}:
        cfg.spread.spread_price = float(cfg.spread.spread_price) * m
        if cfg.spread.spread_points is not None:
            cfg.spread.spread_points = float(cfg.spread.spread_points) * m
    if st in {"COMMISSION", "COMBINED"}:
        cfg.fee.fee_rate_per_side = float(cfg.fee.fee_rate_per_side) * m
        cfg.fee.commission_per_lot = float(cfg.fee.commission_per_lot) * m
    if st in {"SLIPPAGE", "COMBINED"}:
        cfg.slippage.slippage_price = float(cfg.slippage.slippage_price) * m
        cfg.slippage.slippage_rate = float(cfg.slippage.slippage_rate) * m
    return cfg


def _reprice_controlled_commission(
    baseline_result: Any,
    fee_model: BaseFeeModel,
) -> tuple[float, float]:
    """
    Giữ cố định giao dịch, khối lượng và giá khớp từ baseline_result.
    Chỉ tính lại phí qua FeeModel của Group D cho từng Fill ENTRY và EXIT.
    Trả về (total_fees, total_net_pnl_trades).
    """
    trades = getattr(baseline_result, "trades", None)
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return 0.0, 0.0

    total_fees = 0.0
    total_net = 0.0
    for _, row in trades.iterrows():
        qty = float(row.get("quantity", 0.0) or 0.0)
        units = float(row.get("units", 0.0) or 0.0)
        entry_p = float(row.get("entry_price", 0.0) or 0.0)
        exit_p = float(row.get("exit_price", 0.0) or 0.0)
        gross = float(row.get("gross_pnl", 0.0) or 0.0)
        direction = str(row.get("direction", "LONG")).upper()
        entry_side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL
        exit_side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY

        e_fee = fee_model.calculate_fee(entry_p, units, quantity=qty, side=entry_side, is_opening=True)
        x_fee = fee_model.calculate_fee(exit_p, units, quantity=qty, side=exit_side, is_opening=False)
        trade_fee = e_fee + x_fee
        total_fees += trade_fee
        total_net += gross - trade_fee
    return float(total_fees), float(total_net)


def run_controlled_commission_test(
    baseline_result: Any,
    baseline_config: EngineConfig,
    multipliers: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
) -> tuple[pd.DataFrame, bool]:
    """
    Controlled Cost Test: Giữ nguyên giao dịch và giá khớp, chỉ thay đổi Commission.
    Xác minh: khi Commission tăng, Net PnL không được tăng.
    """
    sorted_mults = sorted({float(m) for m in multipliers})
    rows: list[dict[str, Any]] = []
    init_bal = float(baseline_config.account.initial_balance)
    for m in sorted_mults:
        cfg_m = _scale_config(baseline_config, "COMMISSION", m)
        fee_model = BacktestEngine._build_fee_model(cfg_m)
        tot_fees, tot_net = _reprice_controlled_commission(baseline_result, fee_model)
        ret_pct = (tot_net / init_bal) * 100.0 if init_bal > 0.0 else 0.0
        rows.append(
            {
                "multiplier": m,
                "fee_rate_per_side": cfg_m.fee.fee_rate_per_side,
                "commission_per_lot": cfg_m.fee.commission_per_lot,
                "total_fees": tot_fees,
                "net_pnl": tot_net,
                "return_pct": ret_pct,
                "total_trades": len(baseline_result.trades) if isinstance(baseline_result.trades, pd.DataFrame) else 0,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) <= 1:
        return df, True
    pnls = df["net_pnl"].to_numpy(dtype=float)
    is_non_increasing = bool(np.all(np.diff(pnls) <= 1e-9))
    return df, is_non_increasing


def estimate_break_even_multiplier(
    scenarios_df: pd.DataFrame,
    stress_type: str,
) -> float | None:
    """
    Tìm hệ số chi phí hòa vốn NetTradingProfit(m) = 0 bằng nội suy trên khoảng đổi dấu duy nhất.
    Chỉ trả về giá trị khi lợi nhuận không tăng theo m tại khoảng đổi dấu (hợp lý về mặt kinh tế).
    """
    if scenarios_df is None or scenarios_df.empty:
        return None
    sub = scenarios_df[scenarios_df["stress_type"] == stress_type].sort_values("multiplier")
    if len(sub) < 2:
        return None
    ms = sub["multiplier"].to_numpy(dtype=float)
    profits = sub["net_trading_profit"].to_numpy(dtype=float)

    for m, p in zip(ms, profits):
        if abs(p) <= 1e-12:
            return float(m)

    sign_changes = []
    for i in range(len(ms) - 1):
        p0, p1 = profits[i], profits[i + 1]
        if p0 > 0.0 and p1 < 0.0:
            sign_changes.append((ms[i], p0, ms[i + 1], p1))

    if len(sign_changes) != 1:
        return None
    m0, p0, m1, p1 = sign_changes[0]
    frac = p0 / (p0 - p1)
    return float(m0 + frac * (m1 - m0))


def run_friction_stress_test(
    market_data: pd.DataFrame,
    strategy: Any,
    config: EngineConfig | None = None,
    *,
    multipliers: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
    stress_types: Sequence[str] = ("COMBINED",),
    include_controlled_commission: bool = True,
) -> FrictionStressReport:
    """
    Chạy Full Engine Stress Test trên Spread, Commission, Slippage, Combined
    và Controlled Commission Test. Không sửa config gốc hay strategy gốc.
    """
    if market_data is None or market_data.empty:
        return FrictionStressReport(
            status=ValidationStatus.NOT_RUN,
            scenarios_df=pd.DataFrame(),
            messages=["EMPTY_DATA: Không có dữ liệu để chạy Stress Test."],
        )

    base_cfg = clone_config(config)
    baseline_res = run_backtest(market_data, instantiate_fresh_strategy(strategy), base_cfg)
    baseline_metrics = dict(baseline_res.metrics)

    controlled_df = pd.DataFrame()
    is_ctrl_mono = True
    if include_controlled_commission:
        controlled_df, is_ctrl_mono = run_controlled_commission_test(
            baseline_res, base_cfg, multipliers=multipliers
        )

    scenario_rows: list[dict[str, Any]] = []
    for st in stress_types:
        st_upper = str(st).strip().upper()
        for m in multipliers:
            cfg_m = _scale_config(base_cfg, st_upper, float(m))
            res_m = run_backtest(market_data, instantiate_fresh_strategy(strategy), cfg_m)
            met = res_m.metrics
            cf = summarize_ledger_cash_flows(res_m.ledger, initial_balance=cfg_m.account.initial_balance)
            net_profit = float(met.get("net_trading_profit", res_m.final_equity - cfg_m.account.initial_balance - cf.net_external_cash_flow))
            row = StressScenarioResult(
                scenario_name=f"{st_upper}_{m:g}x",
                mode="FULL_ENGINE",
                stress_type=st_upper,
                multiplier=float(m),
                spread_price=float(cfg_m.spread.spread_price),
                spread_points=cfg_m.spread.spread_points,
                fee_rate_per_side=float(cfg_m.fee.fee_rate_per_side),
                commission_per_lot=float(cfg_m.fee.commission_per_lot),
                slippage_price=float(cfg_m.slippage.slippage_price),
                slippage_rate=float(cfg_m.slippage.slippage_rate),
                total_return_pct=float(met.get("total_return_pct", 0.0)),
                net_trading_profit=net_profit,
                max_drawdown_pct=float(met.get("max_drawdown_pct", 0.0)),
                sharpe_annual=float(met.get("sharpe_annual", 0.0)),
                profit_factor=float(met.get("profit_factor", 0.0)),
                total_trades=int(met.get("total_trades", 0) or 0),
                total_fees=float(met.get("total_fees", 0.0) or 0.0),
                total_slippage_cost=float(met.get("total_slippage_cost", 0.0) or 0.0),
            )
            scenario_rows.append(row.__dict__)

    scenarios_df = pd.DataFrame(scenario_rows)
    be_map: dict[str, float | None] = {
        str(st).strip().upper(): estimate_break_even_multiplier(scenarios_df, str(st).strip().upper())
        for st in stress_types
    }
    primary_be = be_map.get("COMBINED")
    if primary_be is None and be_map:
        primary_be = next(iter(be_map.values()))

    messages: list[str] = []
    status = ValidationStatus.PASS
    if not is_ctrl_mono:
        status = ValidationStatus.FAIL
        messages.append("CONTROLLED_COMMISSION_NON_MONOTONIC: Net PnL tăng khi Commission tăng.")
    elif primary_be is not None and primary_be <= 1.0:
        status = ValidationStatus.WARNING
        messages.append(f"LOW_FRICTION_MARGIN: Break-even multiplier={primary_be:.2f}x <= 1.0x.")

    return FrictionStressReport(
        status=status,
        scenarios_df=scenarios_df,
        controlled_commission_df=controlled_df,
        baseline_metrics=baseline_metrics,
        break_even_multipliers=be_map,
        break_even_multiplier=primary_be,
        is_controlled_commission_monotonic=is_ctrl_mono,
        messages=messages,
    )
