from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
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


class Ledger:
    def __init__(self):
        self.entries: list[LedgerEntry] = []

    def record(self, time: pd.Timestamp, transaction_type: TransactionType, amount: float, balance_after: float, description: str = ""):
        entry = LedgerEntry(
            time=time,
            transaction_type=transaction_type,
            amount=float(amount),
            balance_after=float(balance_after),
            description=description,
        )
        self.entries.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["time", "transaction_type", "amount", "balance_after", "description"])
        return pd.DataFrame([
            {
                "time": e.time,
                "transaction_type": e.transaction_type.value,
                "amount": e.amount,
                "balance_after": e.balance_after,
                "description": e.description,
            }
            for e in self.entries
        ])

