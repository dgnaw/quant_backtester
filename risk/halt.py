from __future__ import annotations
from enum import Enum
from typing import Any
import pandas as pd

from .audit import RiskAuditLog


class HaltReason(Enum):
    HALTED_DAILY_LOSS = "HALTED_DAILY_LOSS"
    HALTED_MAX_DRAWDOWN = "HALTED_MAX_DRAWDOWN"
    HALTED_MARGIN_CALL = "HALTED_MARGIN_CALL"
    HALTED_STOP_OUT = "HALTED_STOP_OUT"
    HALTED_MANUAL = "HALTED_MANUAL"
    HALTED_SYSTEM_ERROR = "HALTED_SYSTEM_ERROR"


_HALT_PRIORITY: tuple[tuple[HaltReason, str], ...] = (
    (HaltReason.HALTED_SYSTEM_ERROR, "TRADING_HALTED_SYSTEM_ERROR"),
    (HaltReason.HALTED_MANUAL, "TRADING_HALTED_MANUAL"),
    (HaltReason.HALTED_STOP_OUT, "STOP_OUT_HALT_ACTIVE"),
    (HaltReason.HALTED_MAX_DRAWDOWN, "MAX_DRAWDOWN_LIMIT_REACHED"),
    (HaltReason.HALTED_DAILY_LOSS, "DAILY_LOSS_LIMIT_REACHED"),
    (HaltReason.HALTED_MARGIN_CALL, "MARGIN_CALL_ACTIVE"),
)


class TradingHaltManager:
    """
    Quản lý tập hợp các nguyên nhân khóa giao dịch mở mới (Multi-Reason Trading Halt State Machine).
    """

    def __init__(self, audit_log: RiskAuditLog | None = None) -> None:
        self._active_reasons: set[HaltReason] = set()
        self.audit_log = audit_log

    @property
    def is_halted(self) -> bool:
        return len(self._active_reasons) > 0

    @property
    def active_reasons(self) -> frozenset[HaltReason]:
        return frozenset(self._active_reasons)

    def has_reason(self, reason: HaltReason) -> bool:
        return reason in self._active_reasons

    def trigger_halt(
        self,
        reason: HaltReason,
        timestamp: pd.Timestamp | None = None,
        details: str | dict[str, Any] = "",
    ) -> bool:
        if not isinstance(reason, HaltReason):
            reason = HaltReason(reason)
        if reason in self._active_reasons:
            return False
        self._active_reasons.add(reason)
        if self.audit_log is not None:
            self.audit_log.record(
                event_type="HALT_TRIGGERED",
                status="HALTED",
                timestamp=timestamp,
                reason=reason.value,
                details=details or reason.value,
            )
        return True

    def clear_halt(
        self,
        reason: HaltReason,
        timestamp: pd.Timestamp | None = None,
        details: str | dict[str, Any] = "",
    ) -> bool:
        if not isinstance(reason, HaltReason):
            reason = HaltReason(reason)
        if reason not in self._active_reasons:
            return False
        self._active_reasons.remove(reason)
        if self.audit_log is not None:
            self.audit_log.record(
                event_type="HALT_CLEARED",
                status="ACTIVE" if not self._active_reasons else "HALTED",
                timestamp=timestamp,
                reason=reason.value,
                details=details or f"Cleared {reason.value}",
            )
        return True

    # Aliases for compatibility
    def activate(
        self,
        reason: HaltReason,
        *,
        timestamp: pd.Timestamp | None = None,
        details: dict[str, Any] | str | None = None,
    ) -> bool:
        return self.trigger_halt(reason, timestamp=timestamp, details=details or "")

    def deactivate(
        self,
        reason: HaltReason,
        *,
        timestamp: pd.Timestamp | None = None,
        details: dict[str, Any] | str | None = None,
    ) -> bool:
        return self.clear_halt(reason, timestamp=timestamp, details=details or "")

    @property
    def primary_reject_code(self) -> str | None:
        if not self._active_reasons:
            return None
        for halt_reason, reject_code in _HALT_PRIORITY:
            if halt_reason in self._active_reasons:
                return reject_code
        return next(iter(self._active_reasons)).value

    def primary_rejection_reason(self) -> str | None:
        return self.primary_reject_code

    def reset(self) -> None:
        self._active_reasons.clear()

    def reset_all(self) -> None:
        self._active_reasons.clear()
