from __future__ import annotations
import pandas as pd
from .ledger import Ledger, TransactionType


class Account:
    def __init__(self, initial_balance: float = 1_000.0, base_currency: str = "USD", ledger: Ledger | None = None):
        if initial_balance <= 0:
            raise ValueError("initial_balance phải lớn hơn 0.")
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.base_currency = base_currency
        self.ledger = ledger if ledger is not None else Ledger()
        self.unrealized_pnl = 0.0

    @property
    def equity(self) -> float:
        return self.balance + self.unrealized_pnl

    def deposit(self, amount: float, timestamp: pd.Timestamp, reason: str = "DEPOSIT"):
        if amount <= 0:
            raise ValueError("Số tiền nạp phải > 0.")
        self.balance += float(amount)
        self.ledger.record(timestamp, TransactionType.DEPOSIT, amount, self.balance, reason)

    def withdraw(self, amount: float, timestamp: pd.Timestamp, reason: str = "WITHDRAWAL"):
        if amount <= 0:
            raise ValueError("Số tiền rút phải > 0.")
        if amount > self.balance:
            raise ValueError("Số dư không đủ để rút tiền.")
        self.balance -= float(amount)
        self.ledger.record(timestamp, TransactionType.WITHDRAWAL, -amount, self.balance, reason)

    def deduct_fee(self, fee_amount: float, timestamp: pd.Timestamp, reason: str = "FEE"):
        fee = float(fee_amount)
        if fee > 0:
            self.balance -= fee
            self.ledger.record(timestamp, TransactionType.FEE, -fee, self.balance, reason)

    def realize_trade_pnl(self, gross_pnl: float, exit_fee: float, timestamp: pd.Timestamp, reason: str = "TRADE_PNL"):
        # Entry fee đã được deduct lúc mở vị thế. Lúc đóng chỉ cộng gross_pnl và trừ exit_fee.
        cash_change = float(gross_pnl) - float(exit_fee)
        self.balance += cash_change
        self.ledger.record(timestamp, TransactionType.TRADE_PNL, cash_change, self.balance, reason)

