from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np
import pandas as pd

from quant_backtester.portfolio.ledger import LEDGER_COLUMNS, TransactionType


@dataclass(frozen=True)
class LedgerCashFlowSummary:
    """
    Tổng hợp dòng tiền từ Ledger, tách biệt vốn khởi tạo (Initial Balance)
    với dòng tiền nạp/rút ngoại sinh phát sinh trong kỳ (External Cash Flows).
    """

    initial_balance: float
    total_deposits: float
    total_withdrawals: float
    net_external_cash_flow: float
    total_ledger_fees: float
    total_ledger_gross_pnl: float
    final_ledger_balance: float
    cash_flows_by_time: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass(frozen=True)
class ReportAccountingReconciliation:
    """
    Kết quả đối soát kế toán cấp độ báo cáo (Gate 03 Report Reconciliation).
    """

    is_valid: bool
    expected_value: float
    actual_value: float
    difference: float
    tolerance: float
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


def _is_initial_balance_entry(row: pd.Series | dict[str, Any]) -> bool:
    reason = str(row.get("reason") or row.get("description") or "").strip().upper()
    if reason in {"INITIAL_BALANCE", "INITIAL_DEPOSIT"}:
        return True
    meta = row.get("metadata")
    if isinstance(meta, dict) and bool(meta.get("is_initial_balance")):
        return True
    return False


def summarize_ledger_cash_flows(
    ledger_df: pd.DataFrame | None,
    initial_balance: float = 1_000.0,
) -> LedgerCashFlowSummary:
    """
    Tổng hợp dòng tiền từ bảng Ledger mà không sửa đổi dữ liệu gốc.
    Không dùng ID cứng (như LED-00001) để đoán vốn gốc; chỉ loại trừ bản ghi
    được gắn nhãn tường minh là INITIAL_BALANCE / is_initial_balance.
    """
    if (
        isinstance(initial_balance, bool)
        or not isinstance(initial_balance, (int, float))
        or not math.isfinite(float(initial_balance))
        or float(initial_balance) <= 0.0
    ):
        raise ValueError("initial_balance phải là số thực dương hữu hạn.")
    init_bal = float(initial_balance)

    if ledger_df is None or ledger_df.empty:
        return LedgerCashFlowSummary(
            initial_balance=init_bal,
            total_deposits=0.0,
            total_withdrawals=0.0,
            net_external_cash_flow=0.0,
            total_ledger_fees=0.0,
            total_ledger_gross_pnl=0.0,
            final_ledger_balance=init_bal,
            cash_flows_by_time=pd.Series(dtype=float),
        )

    df = ledger_df.copy()
    time_col = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col is None or "transaction_type" not in df.columns or "amount" not in df.columns:
        raise ValueError(
            f"ledger_df thiếu các cột bắt buộc ('time'/'timestamp', 'transaction_type', 'amount'). Hiện có: {list(df.columns)}"
        )

    total_deposits = 0.0
    total_withdrawals = 0.0
    total_fees = 0.0
    total_gross_pnl = 0.0
    cf_records: list[tuple[pd.Timestamp, float]] = []

    for _, row in df.iterrows():
        tx_raw = row["transaction_type"]
        tx_str = tx_raw.value if isinstance(tx_raw, TransactionType) else str(tx_raw).strip().upper()
        amt = float(row["amount"])
        ts = pd.Timestamp(row[time_col])

        if tx_str == TransactionType.DEPOSIT.value:
            if _is_initial_balance_entry(row):
                continue
            dep_amt = abs(amt)
            total_deposits += dep_amt
            cf_records.append((ts, dep_amt))
        elif tx_str == TransactionType.WITHDRAWAL.value:
            wd_amt = abs(amt)
            total_withdrawals += wd_amt
            cf_records.append((ts, -wd_amt))
        elif tx_str == TransactionType.FEE.value:
            total_fees += abs(amt)
        elif tx_str == TransactionType.TRADE_PNL.value:
            total_gross_pnl += amt

    if cf_records:
        cf_df = pd.DataFrame(cf_records, columns=["time", "cf"])
        cf_series = cf_df.groupby("time", sort=True)["cf"].sum().astype(float)
    else:
        cf_series = pd.Series(dtype=float)

    net_ext_cf = total_deposits - total_withdrawals
    final_led_bal = init_bal + net_ext_cf + total_gross_pnl - total_fees

    return LedgerCashFlowSummary(
        initial_balance=init_bal,
        total_deposits=float(total_deposits),
        total_withdrawals=float(total_withdrawals),
        net_external_cash_flow=float(net_ext_cf),
        total_ledger_fees=float(total_fees),
        total_ledger_gross_pnl=float(total_gross_pnl),
        final_ledger_balance=float(final_led_bal),
        cash_flows_by_time=cf_series,
    )


def map_cash_flows_to_bars(
    bar_times: pd.Series | pd.Index,
    cash_flows_by_time: pd.Series,
) -> np.ndarray:
    """
    Ánh xạ dòng tiền ngoại sinh vào từng Bar theo quy ước cuối Bar (End-of-Bar Cash Flow):
    Mọi dòng tiền tại timestamp ts nằm trong (time_{t-1}, time_t] được gộp vào Bar t.
    """
    n_bars = len(bar_times)
    cf_per_bar = np.zeros(n_bars, dtype=float)
    if n_bars == 0 or cash_flows_by_time is None or cash_flows_by_time.empty:
        return cf_per_bar

    bar_idx_time = pd.DatetimeIndex(pd.to_datetime(bar_times))
    for ts, amount in cash_flows_by_time.items():
        ts_val = pd.Timestamp(ts)
        pos = int(bar_idx_time.searchsorted(ts_val, side="left"))
        if pos >= n_bars:
            pos = n_bars - 1
        cf_per_bar[pos] += float(amount)
    return cf_per_bar


def reconcile_report_accounting(
    result_or_equity: Any = None,
    *,
    equity_df: pd.DataFrame | None = None,
    trades_df: pd.DataFrame | None = None,
    ledger_df: pd.DataFrame | None = None,
    open_positions_df: pd.DataFrame | None = None,
    initial_balance: float | None = None,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-6,
) -> ReportAccountingReconciliation:
    """
    Kiểm tra đối soát kế toán cấp độ báo cáo (Gate 03):
        Actual_Net_Trading_Profit = Final_Equity - Initial_Balance - Net_External_Cash_Flow
        Expected_Net_Trading_Profit =
            Σ(Closed_Trades_Net_PnL)
            + Σ(Open_Positions_Unrealized_PnL)
            - Σ(Open_Positions_Remaining_Entry_Fee)
    """
    if result_or_equity is not None:
        if isinstance(result_or_equity, pd.DataFrame):
            if equity_df is None:
                equity_df = result_or_equity
        elif isinstance(result_or_equity, dict):
            equity_df = equity_df if equity_df is not None else result_or_equity.get("equity")
            trades_df = trades_df if trades_df is not None else result_or_equity.get("trades")
            ledger_df = ledger_df if ledger_df is not None else result_or_equity.get("ledger")
            open_positions_df = (
                open_positions_df
                if open_positions_df is not None
                else result_or_equity.get("open_positions")
            )
            if initial_balance is None:
                cfg = result_or_equity.get("config")
                if cfg is not None and hasattr(cfg, "initial_balance"):
                    initial_balance = float(cfg.initial_balance)
                elif "initial_balance" in result_or_equity:
                    initial_balance = float(result_or_equity["initial_balance"])
        else:
            equity_df = equity_df if equity_df is not None else getattr(result_or_equity, "equity", None)
            trades_df = trades_df if trades_df is not None else getattr(result_or_equity, "trades", None)
            ledger_df = ledger_df if ledger_df is not None else getattr(result_or_equity, "ledger", None)
            open_positions_df = (
                open_positions_df
                if open_positions_df is not None
                else getattr(result_or_equity, "open_positions", None)
            )
            if initial_balance is None:
                cfg = getattr(result_or_equity, "config", None)
                if cfg is not None and hasattr(cfg, "initial_balance"):
                    initial_balance = float(cfg.initial_balance)

    init_bal = float(initial_balance) if initial_balance is not None else 1_000.0
    summary = summarize_ledger_cash_flows(ledger_df, initial_balance=init_bal)

    if equity_df is None or equity_df.empty:
        final_equity = init_bal + summary.net_external_cash_flow
        final_balance: float | None = init_bal + summary.net_external_cash_flow
    else:
        if "equity" not in equity_df.columns:
            raise ValueError("equity_df phải có cột 'equity'.")
        final_equity = float(equity_df["equity"].iloc[-1])
        final_balance = (
            float(equity_df["balance"].iloc[-1]) if "balance" in equity_df.columns else None
        )

    closed_trades_net_pnl = 0.0
    closed_trades_gross_pnl = 0.0
    closed_trades_fees = 0.0
    if trades_df is not None and not trades_df.empty:
        if "net_pnl" in trades_df.columns:
            closed_trades_net_pnl = float(trades_df["net_pnl"].astype(float).sum())
        if "gross_pnl" in trades_df.columns:
            closed_trades_gross_pnl = float(trades_df["gross_pnl"].astype(float).sum())
        if "total_fee" in trades_df.columns:
            closed_trades_fees = float(trades_df["total_fee"].astype(float).sum())
        elif "entry_fee" in trades_df.columns and "exit_fee" in trades_df.columns:
            closed_trades_fees = float(
                (trades_df["entry_fee"].astype(float) + trades_df["exit_fee"].astype(float)).sum()
            )

    open_unrealized_pnl = 0.0
    open_remaining_entry_fee = 0.0
    if open_positions_df is not None and not open_positions_df.empty:
        if "unrealized_pnl" in open_positions_df.columns:
            open_unrealized_pnl = float(open_positions_df["unrealized_pnl"].astype(float).sum())
        if "remaining_entry_fee" in open_positions_df.columns:
            open_remaining_entry_fee = float(
                open_positions_df["remaining_entry_fee"].astype(float).sum()
            )
        elif "entry_fee" in open_positions_df.columns:
            open_remaining_entry_fee = float(open_positions_df["entry_fee"].astype(float).sum())

    actual_net_profit = final_equity - init_bal - summary.net_external_cash_flow
    expected_net_profit = (
        closed_trades_net_pnl + open_unrealized_pnl - open_remaining_entry_fee
    )
    diff_main = actual_net_profit - expected_net_profit

    passed: list[str] = []
    failed: list[str] = []
    messages: list[str] = []

    def _verify(check_name: str, actual_v: float, expected_v: float, desc: str) -> None:
        if math.isclose(actual_v, expected_v, rel_tol=rel_tol, abs_tol=abs_tol):
            passed.append(check_name)
        else:
            failed.append(check_name)
            messages.append(
                f"[{check_name}] {desc}: actual={actual_v:.8f}, expected={expected_v:.8f}, diff={actual_v - expected_v:.8e}"
            )

    _verify(
        "REPORT_NET_PROFIT_CONSERVATION",
        actual_net_profit,
        expected_net_profit,
        "Lợi nhuận ròng từ Equity lệch với tổng PnL từ Closed Trades + Open Positions",
    )

    if final_balance is not None:
        _verify(
            "EQUITY_MARK_TO_MARKET_MATCH",
            final_equity,
            final_balance + open_unrealized_pnl,
            "Final Equity lệch Final Balance + Open Unrealized PnL",
        )

    if ledger_df is not None and not ledger_df.empty:
        if final_balance is not None:
            _verify(
                "LEDGER_FINAL_BALANCE_MATCH",
                final_balance,
                summary.final_ledger_balance,
                "Final Balance trên Equity Curve lệch với số dư tính từ Ledger",
            )
        ledger_net_realized_impact = summary.total_ledger_gross_pnl - summary.total_ledger_fees
        trades_and_open_fee_impact = closed_trades_net_pnl - open_remaining_entry_fee
        _verify(
            "TRADES_VS_LEDGER_NET_PNL",
            ledger_net_realized_impact,
            trades_and_open_fee_impact,
            "Tổng PnL và phí trên Ledger lệch với Closed Trades Net PnL trừ Remaining Entry Fee",
        )

    details = {
        "initial_balance": init_bal,
        "final_equity": final_equity,
        "final_balance": final_balance if final_balance is not None else float("nan"),
        "total_deposits": summary.total_deposits,
        "total_withdrawals": summary.total_withdrawals,
        "net_external_cash_flow": summary.net_external_cash_flow,
        "closed_trades_net_pnl": closed_trades_net_pnl,
        "closed_trades_gross_pnl": closed_trades_gross_pnl,
        "closed_trades_fees": closed_trades_fees,
        "open_unrealized_pnl": open_unrealized_pnl,
        "open_remaining_entry_fee": open_remaining_entry_fee,
        "actual_net_trading_profit": actual_net_profit,
        "expected_net_trading_profit": expected_net_profit,
    }

    eff_tol = max(abs_tol, rel_tol * max(abs(actual_net_profit), abs(expected_net_profit), 1.0))
    return ReportAccountingReconciliation(
        is_valid=(len(failed) == 0),
        expected_value=float(expected_net_profit),
        actual_value=float(actual_net_profit),
        difference=float(diff_main),
        tolerance=float(eff_tol),
        checks_passed=passed,
        checks_failed=failed,
        messages=messages,
        details=details,
    )
