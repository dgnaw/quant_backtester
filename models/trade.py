from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class EntryContribution:
    """
    Truy vết một lần khớp lệnh mở (Entry Fill) cấu thành nên Position hiện tại.
    """
    order_id: str | None
    fill_id: str | None
    time: pd.Timestamp
    bar_index: int | None
    price: float
    initial_units: float
    remaining_units: float
    initial_entry_fee: float = 0.0
    remaining_entry_fee: float = 0.0
    initial_slippage_cost: float = 0.0
    remaining_slippage_cost: float = 0.0
    contract_size: float = 100.0
    initial_stop_loss_price: float | None = None
    initial_risk_amount: float | None = None
    remaining_risk_amount: float | None = None

    @property
    def initial_quantity(self) -> float:
        return self.initial_units / self.contract_size if self.contract_size > 0 else 0.0

    @property
    def remaining_quantity(self) -> float:
        return self.remaining_units / self.contract_size if self.contract_size > 0 else 0.0


@dataclass(frozen=True)
class TradeAllocation:
    """
    Bản ghi phân bổ một phần (hoặc toàn bộ) EntryContribution vào một TradeRecord khi đóng vị thế.
    """
    trade_id: str
    source_entry_order_id: str | None
    source_entry_fill_id: str | None
    source_entry_time: pd.Timestamp
    source_entry_bar_index: int | None
    allocated_units: float
    allocated_quantity: float
    allocated_entry_fee: float
    allocated_entry_slippage_cost: float
    allocated_risk_amount: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "source_entry_order_id": self.source_entry_order_id,
            "source_entry_fill_id": self.source_entry_fill_id,
            "source_entry_time": self.source_entry_time,
            "source_entry_bar_index": self.source_entry_bar_index,
            "allocated_units": self.allocated_units,
            "allocated_quantity": self.allocated_quantity,
            "allocated_entry_fee": self.allocated_entry_fee,
            "allocated_entry_slippage_cost": self.allocated_entry_slippage_cost,
            "allocated_risk_amount": self.allocated_risk_amount,
        }


@dataclass
class TradeRecord:
    """
    Ghi nhận kết quả chuẩn hóa của một phần hoặc toàn bộ Position đã được đóng.
    """
    trade_id: str
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    units: float
    entry_fee: float
    exit_fee: float
    total_fee: float
    entry_slippage_cost: float
    exit_slippage_cost: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    reason: str
    position_id: str | None = None
    entry_bar_index: int | None = None
    exit_bar_index: int | None = None
    holding_bars: int | float | None = None
    weighted_holding_bars: float | None = None
    entry_order_ids: list[str] = field(default_factory=list)
    entry_fill_ids: list[str] = field(default_factory=list)
    entry_order_id: str | None = None
    entry_fill_id: str | None = None
    exit_order_id: str | None = None
    exit_fill_id: str | None = None
    initial_risk_amount: float | None = None
    r_multiple: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_bar_index": self.entry_bar_index,
            "exit_bar_index": self.exit_bar_index,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_order_ids": list(self.entry_order_ids),
            "entry_fill_ids": list(self.entry_fill_ids),
            "entry_order_id": self.entry_order_id,
            "entry_fill_id": self.entry_fill_id,
            "exit_order_id": self.exit_order_id,
            "exit_fill_id": self.exit_fill_id,
            "quantity": self.quantity,
            "units": self.units,
            "holding_bars": self.holding_bars,
            "weighted_holding_bars": self.weighted_holding_bars,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "total_fee": self.total_fee,
            "entry_slippage_cost": self.entry_slippage_cost,
            "exit_slippage_cost": self.exit_slippage_cost,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "initial_risk_amount": self.initial_risk_amount,
            "r_multiple": self.r_multiple,
            "reason": self.reason,
        }


@dataclass
class ReconciliationReport:
    """
    Báo cáo kiểm tra tự động 7 bất biến kế toán (Accounting Invariants) giữa Account, Portfolio và Ledger.
    """
    is_valid: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    differences: dict[str, float] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return self.messages

    @property
    def cash_invariant_ok(self) -> bool:
        return "INVARIANT_1_BALANCE" not in self.checks_failed

    @property
    def realized_pnl_conservation_ok(self) -> bool:
        return (
            "INVARIANT_2A_REALIZED_GROSS_TRADES" not in self.checks_failed
            and "INVARIANT_2B_REALIZED_GROSS_LEDGER" not in self.checks_failed
        )

    @property
    def fee_conservation_ok(self) -> bool:
        return (
            "INVARIANT_3A_FEE_SPLIT" not in self.checks_failed
            and "INVARIANT_3B_FEE_POSITIONS_TRADES" not in self.checks_failed
            and "INVARIANT_3C_FEE_LEDGER" not in self.checks_failed
        )

    @property
    def ledger_balance_ok(self) -> bool:
        return "INVARIANT_4_LEDGER_RUNNING_BALANCE" not in self.checks_failed

    @property
    def equity_invariant_ok(self) -> bool:
        return "INVARIANT_5_EQUITY" not in self.checks_failed

    @property
    def position_allocation_conservation_ok(self) -> bool:
        return "INVARIANT_6_ENTRY_CONTRIBUTION" not in self.checks_failed

    @property
    def fill_idempotency_ok(self) -> bool:
        return "INVARIANT_7_FILL_IDEMPOTENCY" not in self.checks_failed
