from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import pandas as pd


class TransactionType(Enum):
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    FEE = "FEE"
    TRADE_PNL = "TRADE_PNL"


@dataclass
class LedgerEntry:
    time: pd.Timestamp
    transaction_type: TransactionType
    amount: float
    balance_after: float
    description: str = ""
    entry_id: str = ""
    symbol: str | None = None
    order_id: str | None = None
    fill_id: str | None = None
    trade_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp(self) -> pd.Timestamp:
        return self.time

    @property
    def reason(self) -> str:
        return self.description


LEDGER_COLUMNS = [
    "entry_id",
    "time",
    "timestamp",
    "transaction_type",
    "amount",
    "balance_after",
    "symbol",
    "order_id",
    "fill_id",
    "trade_id",
    "reason",
    "description",
]


class Ledger:
    def __init__(self):
        self.entries: list[LedgerEntry] = []
        self._entry_counter = 0

    def _next_entry_id(self) -> str:
        self._entry_counter += 1
        return f"LED-{self._entry_counter:05d}"

    def record(
        self,
        time: pd.Timestamp,
        transaction_type: TransactionType,
        amount: float,
        balance_after: float,
        description: str = "",
        *,
        symbol: str | None = None,
        order_id: str | None = None,
        fill_id: str | None = None,
        trade_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        eff_desc = reason if reason is not None else description
        eid = self._next_entry_id()
        ts = pd.Timestamp(time)
        entry = LedgerEntry(
            entry_id=eid,
            time=ts,
            transaction_type=transaction_type,
            amount=float(amount),
            balance_after=float(balance_after),
            description=eff_desc,
            symbol=symbol,
            order_id=order_id,
            fill_id=fill_id,
            trade_id=trade_id,
            metadata=dict(metadata) if metadata else {},
        )
        self.entries.append(entry)
        return entry

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        return pd.DataFrame([
            {
                "entry_id": e.entry_id,
                "time": e.time,
                "timestamp": e.timestamp,
                "transaction_type": e.transaction_type.value,
                "amount": e.amount,
                "balance_after": e.balance_after,
                "symbol": e.symbol,
                "order_id": e.order_id,
                "fill_id": e.fill_id,
                "trade_id": e.trade_id,
                "reason": e.reason,
                "description": e.description,
            }
            for e in self.entries
        ])
