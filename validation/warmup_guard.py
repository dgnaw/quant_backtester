"""H2 – Warm-up Boundary Guard & Evaluation Window Isolation."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_backtester.analytics.report import calculate_metrics
from quant_backtester.validation.models import ValidationStatus

_TRADING_LEDGER_TYPES = {"FEE", "TRADE_PNL", "COMMISSION", "SWAP"}
_TOL = 1e-9


# ---------------------------------------------------------------------------
# Warm-up audit
# ---------------------------------------------------------------------------
@dataclass
class WarmupAuditReport:
    status: ValidationStatus
    warmup_period: int
    warmup_start_index: int
    signal_start_index: int
    trade_start_index: int
    first_allowed_time: pd.Timestamp | None
    first_fill_time: pd.Timestamp | None
    first_fill_bar_index: int | None
    strict_signal_start: bool
    violations: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.FAIL


def resolve_boundaries(trade_start_index: int, strict_signal_start: bool = False) -> tuple[int, int, int]:
    """
    Trả về (warmup_start_index, signal_start_index, trade_start_index).
    Quy ước Engine: tín hiệu đóng nến (trade_start - 1) được khớp sớm nhất tại Open nến trade_start.
    Chế độ strict: tín hiệu chỉ được phát từ trade_start.
    """
    ts = max(0, int(trade_start_index))
    if strict_signal_start:
        return 0, ts, ts
    return 0, max(0, ts - 1), ts


def _events(result: Any, event_type: str) -> pd.DataFrame:
    ev = getattr(result, "order_events", None)
    if ev is None or not isinstance(ev, pd.DataFrame) or ev.empty or "event_type" not in ev.columns:
        return pd.DataFrame(columns=["order_id", "event_type", "timestamp", "bar_index"])
    return ev[ev["event_type"] == event_type]


def verify_warmup_boundary(
    result: Any,
    strategy: Any = None,
    strict_signal_start: bool = False,
    initial_balance: float | None = None,
) -> WarmupAuditReport:
    """
    Kiểm toán ranh giới warm-up trên BacktestResult (chỉ đọc):
    - Không Fill / Trade Entry / Position / thay đổi Balance-Equity trước trade_start.
    - Không có Order được tạo trước signal_start.
    """
    equity = getattr(result, "equity", None)
    if equity is None or not isinstance(equity, pd.DataFrame) or equity.empty:
        return WarmupAuditReport(
            status=ValidationStatus.NOT_RUN,
            warmup_period=0,
            warmup_start_index=0,
            signal_start_index=0,
            trade_start_index=0,
            first_allowed_time=None,
            first_fill_time=None,
            first_fill_bar_index=None,
            strict_signal_start=strict_signal_start,
            messages=["EMPTY_EQUITY: Không có dữ liệu equity để kiểm toán."],
        )

    trade_start_idx = int(getattr(result, "effective_trade_start", 0) or 0)
    warmup_period = int(getattr(strategy, "warmup_period", trade_start_idx)) if strategy is not None else trade_start_idx
    w_start, sig_start, trade_start_idx = resolve_boundaries(trade_start_idx, strict_signal_start)
    times = pd.to_datetime(equity["time"]).reset_index(drop=True)
    n = len(equity)
    first_allowed_time = times.iloc[trade_start_idx] if trade_start_idx < n else None

    cfg = getattr(result, "config", None)
    init_bal = (
        float(initial_balance)
        if initial_balance is not None
        else float(cfg.account.initial_balance) if cfg is not None else float(equity["equity"].iloc[0])
    )

    violations: list[str] = []
    messages: list[str] = []

    if strategy is not None and warmup_period > trade_start_idx:
        violations.append(
            f"TRADE_START_BEFORE_WARMUP: trade_start_index={trade_start_idx} < warmup_period={warmup_period}."
        )

    fills = _events(result, "FILLED")
    first_fill_bar = int(fills["bar_index"].min()) if not fills.empty else None
    first_fill_time = pd.Timestamp(fills.sort_values("bar_index")["timestamp"].iloc[0]) if not fills.empty else None
    early_fills = fills[fills["bar_index"] < trade_start_idx] if not fills.empty else fills
    for _, row in early_fills.iterrows():
        violations.append(f"FILL_IN_WARMUP: order {row['order_id']} khớp tại bar {int(row['bar_index'])} < {trade_start_idx}.")

    created = _events(result, "CREATED")
    early_orders = created[created["bar_index"] < sig_start] if not created.empty else created
    for _, row in early_orders.iterrows():
        violations.append(
            f"ORDER_BEFORE_SIGNAL_START: order {row['order_id']} tạo tại bar {int(row['bar_index'])} < {sig_start}."
        )

    trades = getattr(result, "trades", None)
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        if "entry_bar_index" in trades.columns:
            bad = trades[pd.to_numeric(trades["entry_bar_index"], errors="coerce") < trade_start_idx]
        elif "entry_time" in trades.columns and first_allowed_time is not None:
            bad = trades[pd.to_datetime(trades["entry_time"]) < first_allowed_time]
        else:
            bad = trades.iloc[0:0]
        for _, row in bad.iterrows():
            violations.append(f"TRADE_ENTRY_IN_WARMUP: trade {row.get('trade_id')} vào lệnh trước trade_start.")

    pre = equity.iloc[:trade_start_idx]
    if not pre.empty:
        bal_dev = np.abs(pre["balance"].astype(float) - init_bal) > _TOL if "balance" in pre.columns else None
        eq_dev = np.abs(pre["equity"].astype(float) - init_bal) > _TOL
        pos_dev = (pre["position"].astype(float) != 0) if "position" in pre.columns else None
        if bal_dev is not None and bal_dev.any():
            violations.append(f"BALANCE_CHANGED_IN_WARMUP: {int(bal_dev.sum())} nến có balance != initial_balance.")
        if eq_dev.any():
            violations.append(f"EQUITY_CHANGED_IN_WARMUP: {int(eq_dev.sum())} nến có equity != initial_balance.")
        if pos_dev is not None and pos_dev.any():
            violations.append(f"POSITION_IN_WARMUP: {int(pos_dev.sum())} nến có position != 0.")

    ledger = getattr(result, "ledger", None)
    if isinstance(ledger, pd.DataFrame) and not ledger.empty and first_allowed_time is not None:
        led_t = pd.to_datetime(ledger["time"])
        early_led = ledger[(led_t < first_allowed_time) & ledger["transaction_type"].astype(str).isin(_TRADING_LEDGER_TYPES)]
        if not early_led.empty:
            violations.append(f"LEDGER_ACTIVITY_IN_WARMUP: {len(early_led)} bút toán giao dịch trước trade_start.")
        ext = ledger[(led_t < first_allowed_time) & ledger["transaction_type"].astype(str).isin({"DEPOSIT", "WITHDRAWAL"})]
        if not ext.empty:
            messages.append("EXTERNAL_CASH_FLOW_IN_WARMUP: có dòng tiền ngoại sinh trong warm-up, giả định bất biến balance có thể không áp dụng.")

    status = ValidationStatus.FAIL if violations else (ValidationStatus.WARNING if messages else ValidationStatus.PASS)
    return WarmupAuditReport(
        status=status,
        warmup_period=warmup_period,
        warmup_start_index=w_start,
        signal_start_index=sig_start,
        trade_start_index=trade_start_idx,
        first_allowed_time=first_allowed_time,
        first_fill_time=first_fill_time,
        first_fill_bar_index=first_fill_bar,
        strict_signal_start=strict_signal_start,
        violations=violations,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Evaluation window
# ---------------------------------------------------------------------------
@dataclass
class EvaluationWindow:
    evaluation_equity_df: pd.DataFrame
    anchored_equity_df: pd.DataFrame
    evaluation_ledger_df: pd.DataFrame
    evaluation_trades_df: pd.DataFrame
    evaluation_start_time: pd.Timestamp | None
    evaluation_end_time: pd.Timestamp | None
    evaluation_start_index: int
    evaluation_bars: int
    warmup_bars: int
    excluded_bars: int
    anchor_equity: float
    anchor_balance: float
    anchor_time: pd.Timestamp | None
    synthetic_anchor: bool

    @property
    def return_observations(self) -> int:
        return max(0, len(self.anchored_equity_df) - 1)


def _synthetic_anchor_time(times: pd.Series) -> pd.Timestamp:
    if len(times) >= 2:
        return pd.Timestamp(times.iloc[0]) - (pd.Timestamp(times.iloc[1]) - pd.Timestamp(times.iloc[0]))
    return pd.Timestamp(times.iloc[0]) - pd.Timedelta(nanoseconds=1)


def extract_evaluation_window(
    result: Any,
    start_index: int | None = None,
    end_index: int | None = None,
    initial_balance: float | None = None,
) -> EvaluationWindow:
    """
    Cắt cửa sổ đánh giá [start_index, end_index) khỏi BacktestResult.
    - anchored_equity_df = 1 dòng neo (equity thực tế ngay trước cửa sổ) + M dòng đánh giá,
      để chuỗi lợi suất có đúng M quan sát và không chứa return giả bằng 0 của warm-up.
    - Khi start_index == 0, dùng điểm neo tổng hợp = initial_balance (synthetic_anchor=True).
    - Cash flow / bút toán Ledger thuộc cửa sổ được giữ nguyên.
    """
    equity = result.equity if hasattr(result, "equity") else result["equity"]
    if equity is None or equity.empty:
        raise ValueError("Equity curve rỗng, không thể trích xuất Evaluation Window.")
    eq = equity.reset_index(drop=True)
    n = len(eq)
    start = int(getattr(result, "effective_trade_start", 0) or 0) if start_index is None else int(start_index)
    end = n if end_index is None else int(end_index)
    if not (0 <= start < end <= n):
        raise ValueError(f"Evaluation window không hợp lệ: start={start}, end={end}, n={n}.")

    cfg = getattr(result, "config", None)
    init_bal = (
        float(initial_balance)
        if initial_balance is not None
        else float(cfg.account.initial_balance) if cfg is not None else float(eq["equity"].iloc[0])
    )

    window = eq.iloc[start:end].reset_index(drop=True)
    if start > 0:
        anchor_row = eq.iloc[[start - 1]].copy()
        synthetic = False
    else:
        anchor_row = pd.DataFrame([{c: window[c].iloc[0] for c in window.columns}])
        anchor_row["time"] = _synthetic_anchor_time(pd.to_datetime(window["time"]))
        anchor_row["equity"] = init_bal
        if "balance" in anchor_row.columns:
            anchor_row["balance"] = init_bal
        if "position" in anchor_row.columns:
            anchor_row["position"] = 0
        synthetic = True
    anchored = pd.concat([anchor_row, window], ignore_index=True)
    anchor_equity = float(anchor_row["equity"].iloc[0])
    anchor_balance = float(anchor_row["balance"].iloc[0]) if "balance" in anchor_row.columns else anchor_equity
    anchor_time = None if synthetic else pd.Timestamp(anchor_row["time"].iloc[0])

    start_time = pd.Timestamp(window["time"].iloc[0])
    end_time = pd.Timestamp(window["time"].iloc[-1])

    ledger = getattr(result, "ledger", None)
    if isinstance(ledger, pd.DataFrame) and not ledger.empty and "time" in ledger.columns:
        lt = pd.to_datetime(ledger["time"])
        mask = (lt >= start_time) & (lt <= end_time)
        eval_ledger = ledger[mask].reset_index(drop=True)
    else:
        eval_ledger = ledger.copy() if isinstance(ledger, pd.DataFrame) else pd.DataFrame()

    trades = getattr(result, "trades", None)
    if isinstance(trades, pd.DataFrame) and not trades.empty and "exit_time" in trades.columns:
        xt = pd.to_datetime(trades["exit_time"])
        eval_trades = trades[(xt >= start_time) & (xt <= end_time)].reset_index(drop=True)
    else:
        eval_trades = trades.copy() if isinstance(trades, pd.DataFrame) else pd.DataFrame()

    warmup_bars = int(getattr(result, "effective_trade_start", 0) or 0)
    return EvaluationWindow(
        evaluation_equity_df=window,
        anchored_equity_df=anchored,
        evaluation_ledger_df=eval_ledger,
        evaluation_trades_df=eval_trades,
        evaluation_start_time=start_time,
        evaluation_end_time=end_time,
        evaluation_start_index=start,
        evaluation_bars=len(window),
        warmup_bars=warmup_bars,
        excluded_bars=n - len(window),
        anchor_equity=anchor_equity,
        anchor_balance=anchor_balance,
        anchor_time=anchor_time,
        synthetic_anchor=synthetic,
    )


def compute_evaluation_metrics(
    result: Any,
    window: EvaluationWindow | None = None,
    config: Any = None,
) -> tuple[dict[str, Any], EvaluationWindow]:
    """
    Tính metrics Group G trên cửa sổ đánh giá (tái sử dụng Analytics, không tự tính lại PnL).
    initial_balance của báo cáo = anchor_equity.
    """
    win = window if window is not None else extract_evaluation_window(result)
    base_cfg = config if config is not None else getattr(result, "config", None)
    cfg = copy.deepcopy(base_cfg)
    if cfg is not None and hasattr(cfg, "account"):
        cfg.account.initial_balance = float(win.anchor_equity)
    payload = {
        "equity": win.anchored_equity_df,
        "trades": win.evaluation_trades_df,
        "ledger": win.evaluation_ledger_df,
        "open_positions": getattr(result, "open_positions", None),
    }
    metrics = calculate_metrics(payload, cfg)
    metrics["evaluation_bars"] = win.evaluation_bars
    metrics["evaluation_return_observations"] = win.return_observations
    metrics["anchor_equity"] = win.anchor_equity
    return metrics, win
