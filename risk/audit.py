from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import pandas as pd

if TYPE_CHECKING:
    from .decision import RiskDecision


RISK_AUDIT_COLUMNS = [
    "audit_event_id",
    "timestamp",
    "event_type",
    "symbol",
    "side",
    "status",
    "reject_code",
    "reason",
    "sized_quantity",
    "quantity",
    "estimated_risk_amount",
    "estimated_risk",
    "required_margin",
    "details",
]


@dataclass(frozen=True)
class RiskAuditEntry:
    audit_event_id: str
    timestamp: pd.Timestamp | None
    event_type: str
    symbol: str | None
    status: str
    side: str | None = None
    reason: str | None = None
    quantity: float | None = None
    estimated_risk: float | None = None
    required_margin: float | None = None
    details: dict[str, Any] | str = field(default_factory=dict)

    @property
    def reject_code(self) -> str | None:
        return self.reason

    @property
    def sized_quantity(self) -> float | None:
        return self.quantity

    @property
    def estimated_risk_amount(self) -> float | None:
        return self.estimated_risk

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_event_id": self.audit_event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "reject_code": self.reason,
            "reason": self.reason,
            "sized_quantity": self.quantity,
            "quantity": self.quantity,
            "estimated_risk_amount": self.estimated_risk,
            "estimated_risk": self.estimated_risk,
            "required_margin": self.required_margin,
            "details": self.details if isinstance(self.details, str) else dict(self.details),
        }


class RiskAuditLog:
    """
    Nhật ký kiểm toán mọi quyết định duyệt/từ chối lệnh và các sự kiện Circuit Breaker / Halt.
    """

    def __init__(self) -> None:
        self.entries: list[RiskAuditEntry] = []
        self._counter: int = 0

    def next_event_id(self) -> str:
        self._counter += 1
        return f"RSK-{self._counter:05d}"

    def record(
        self,
        event_type: str,
        status: str = "INFO",
        *,
        timestamp: pd.Timestamp | None = None,
        symbol: str | None = None,
        side: str | None = None,
        reason: str | None = None,
        quantity: float | None = None,
        estimated_risk: float | None = None,
        required_margin: float | None = None,
        details: dict[str, Any] | str | None = None,
        audit_event_id: str | None = None,
    ) -> RiskAuditEntry:
        eid = audit_event_id if audit_event_id else self.next_event_id()
        ts = pd.Timestamp(timestamp) if timestamp is not None else None
        entry = RiskAuditEntry(
            audit_event_id=eid,
            timestamp=ts,
            event_type=str(event_type),
            symbol=symbol,
            side=side,
            status=str(status),
            reason=reason,
            quantity=float(quantity) if quantity is not None else None,
            estimated_risk=float(estimated_risk) if estimated_risk is not None else None,
            required_margin=float(required_margin) if required_margin is not None else None,
            details=details if details is not None else {},
        )
        self.entries.append(entry)
        return entry

    def record_event(
        self,
        event_type: str,
        timestamp: pd.Timestamp | None = None,
        symbol: str | None = None,
        side: str | None = None,
        reject_code: str | None = None,
        sized_quantity: float | None = None,
        estimated_risk_amount: float | None = None,
        required_margin: float | None = None,
        details: str | dict[str, Any] = "",
    ) -> RiskAuditEntry:
        return self.record(
            event_type=event_type,
            status=event_type,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            reason=reject_code,
            quantity=sized_quantity,
            estimated_risk=estimated_risk_amount,
            required_margin=required_margin,
            details=details,
        )

    def record_decision(
        self,
        timestamp: pd.Timestamp | None,
        symbol: str,
        side: str,
        decision: RiskDecision,
    ) -> RiskAuditEntry:
        event_type = "ORDER_APPROVED" if decision.is_approved else "ORDER_REJECTED"
        failed_msgs = [
            f"{c.check_name}:{c.reject_code}"
            for c in decision.checks
            if not c.passed and c.reject_code
        ]
        return self.record(
            event_type=event_type,
            status=decision.status.value,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            reason=decision.reject_reason,
            quantity=decision.sized_quantity,
            estimated_risk=decision.estimated_risk_amount,
            required_margin=decision.required_margin,
            details="; ".join(failed_msgs) if failed_msgs else "OK",
        )

    def clear(self) -> None:
        self.entries.clear()
        self._counter = 0

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=RISK_AUDIT_COLUMNS)
        return pd.DataFrame([e.to_dict() for e in self.entries], columns=RISK_AUDIT_COLUMNS)
