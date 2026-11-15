from __future__ import annotations
import math
from typing import Any
import pandas as pd
from .ledger import Ledger, LedgerEntry, TransactionType


class Account:
    def __init__(
        self,
        initial_balance: float = 1_000.0,
        base_currency: str = "USD",
        ledger: Ledger | None = None,
        account_currency: str | None = None,
    ):
        if isinstance(initial_balance, bool) or not isinstance(initial_balance, (int, float)) or not math.isfinite(float(initial_balance)) or initial_balance <= 0:
            raise ValueError("initial_balance phải lớn hơn 0 và hữu hạn.")
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        eff_currency = str(account_currency) if account_currency is not None else str(base_currency)
        self.base_currency = eff_currency
        self.ledger = ledger if ledger is not None else Ledger()

        self.net_deposits: float = 0.0
        self.realized_gross_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        self.realized_fees: float = 0.0
        self.open_position_entry_fees: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.used_margin: float = 0.0

    @property
    def account_currency(self) -> str:
        return self.base_currency

    @property
    def total_fees(self) -> float:
        return self.total_fees_paid

    @property
    def realized_net_pnl(self) -> float:
        return self.realized_gross_pnl - self.realized_fees

    @property
    def equity(self) -> float:
        return self.balance + self.unrealized_pnl

    @property
    def free_margin(self) -> float:
        return self.equity - self.used_margin

    @property
    def margin_level_pct(self) -> float:
        if self.used_margin <= 0.0:
            return float("inf")
        return (self.equity / self.used_margin) * 100.0

    def validate_currency_compatibility(self, quote_currency: str) -> None:
        if str(quote_currency).strip().upper() != self.account_currency.strip().upper():
            raise ValueError(
                f"UNSUPPORTED_CURRENCY_CONVERSION: quote_currency ({quote_currency}) khác account_currency ({self.account_currency}). "
                "V1.0 chưa hỗ trợ quy đổi tỷ giá tự động."
            )

    def deposit(
        self,
        amount: float,
        timestamp: pd.Timestamp,
        reason: str = "DEPOSIT",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(float(amount)) or amount <= 0:
            raise ValueError("Số tiền nạp phải > 0.")
        amt = float(amount)
        self.net_deposits += amt
        self.balance += amt
        return self.ledger.record(
            timestamp,
            TransactionType.DEPOSIT,
            amt,
            self.balance,
            description=reason,
            metadata=metadata,
        )

    def withdraw(
        self,
        amount: float,
        timestamp: pd.Timestamp,
        reason: str = "WITHDRAWAL",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(float(amount)) or amount <= 0:
            raise ValueError("Số tiền rút phải > 0.")
        amt = float(amount)
        if amt > self.balance:
            raise ValueError("Số dư không đủ để rút tiền.")
        self.net_deposits -= amt
        self.balance -= amt
        return self.ledger.record(
            timestamp,
            TransactionType.WITHDRAWAL,
            -amt,
            self.balance,
            description=reason,
            metadata=metadata,
        )

    def deduct_fee(
        self,
        fee_amount: float,
        timestamp: pd.Timestamp,
        reason: str = "FEE",
        *,
        is_opening: bool = False,
        symbol: str | None = None,
        order_id: str | None = None,
        fill_id: str | None = None,
        trade_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry | None:
        if isinstance(fee_amount, bool) or not isinstance(fee_amount, (int, float)) or not math.isfinite(float(fee_amount)) or fee_amount < 0:
            raise ValueError("Phí giao dịch không hợp lệ.")
        fee = float(fee_amount)
        if is_opening:
            self.open_position_entry_fees += fee
        else:
            self.realized_fees += fee

        if fee > 0.0:
            self.total_fees_paid += fee
            self.balance -= fee
            return self.ledger.record(
                timestamp,
                TransactionType.FEE,
                -fee,
                self.balance,
                description=reason,
                symbol=symbol,
                order_id=order_id,
                fill_id=fill_id,
                trade_id=trade_id,
                metadata=metadata,
            )
        return None

    def allocate_open_entry_fee_to_trade(self, allocated_entry_fee: float) -> None:
        alloc = float(allocated_entry_fee)
        if alloc > 0.0:
            self.open_position_entry_fees = max(0.0, self.open_position_entry_fees - alloc)
            self.realized_fees += alloc

    def record_realized_gross_pnl(
        self,
        gross_pnl: float,
        timestamp: pd.Timestamp,
        reason: str = "TRADE_PNL",
        *,
        symbol: str | None = None,
        order_id: str | None = None,
        fill_id: str | None = None,
        trade_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        gpnl = float(gross_pnl)
        self.realized_gross_pnl += gpnl
        self.balance += gpnl
        return self.ledger.record(
            timestamp,
            TransactionType.TRADE_PNL,
            gpnl,
            self.balance,
            description=reason,
            symbol=symbol,
            order_id=order_id,
            fill_id=fill_id,
            trade_id=trade_id,
            metadata=metadata,
        )

    def realize_trade_pnl(
        self,
        gross_pnl: float,
        exit_fee: float,
        timestamp: pd.Timestamp,
        reason: str = "TRADE_PNL",
        *,
        symbol: str | None = None,
        order_id: str | None = None,
        fill_id: str | None = None,
        trade_id: str | None = None,
        allocated_entry_fee: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        """
        Ghi nhận kết quả đóng vị thế vào Account & Ledger:
        1. Phân bổ phần Entry Fee đã đóng từ open_position_entry_fees sang realized_fees (không ghi trùng Ledger FEE).
        2. Nếu exit_fee > 0: ghi 1 bút toán TransactionType.FEE riêng cho exit_fee.
        3. Ghi 1 bút toán TransactionType.TRADE_PNL phản ánh gross_pnl.
        """
        if allocated_entry_fee > 0.0:
            self.allocate_open_entry_fee_to_trade(allocated_entry_fee)
        if float(exit_fee) > 0.0:
            self.deduct_fee(
                exit_fee,
                timestamp,
                reason=f"EXIT_FEE_{symbol}" if symbol else "EXIT_FEE",
                is_opening=False,
                symbol=symbol,
                order_id=order_id,
                fill_id=fill_id,
                trade_id=trade_id,
                metadata=metadata,
            )
        return self.record_realized_gross_pnl(
            gross_pnl,
            timestamp,
            reason=reason,
            symbol=symbol,
            order_id=order_id,
            fill_id=fill_id,
            trade_id=trade_id,
            metadata=metadata,
        )
