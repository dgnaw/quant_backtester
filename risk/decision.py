from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from quant_backtester.models.order import OrderIntent

if TYPE_CHECKING:
    from .estimator import RiskEstimate


class RiskDecisionStatus(Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class RiskCheckResult:
    """
    Kết quả đánh giá của một quy tắc rủi ro đơn lẻ (Rule Evaluation).
    """

    check_name: str = ""
    passed: bool = True
    reject_code: str | None = None
    message: str = ""
    current_value: Any = None
    limit_value: Any = None
    rule_name: str | None = None
    actual_value: Any = None
    threshold_value: Any = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        eff_name = self.check_name or self.rule_name or "RISK_CHECK"
        eff_reject = self.reject_code if self.reject_code is not None else self.rejection_reason
        eff_cur = self.current_value if self.current_value is not None else self.actual_value
        eff_lim = self.limit_value if self.limit_value is not None else self.threshold_value
        object.__setattr__(self, "check_name", eff_name)
        object.__setattr__(self, "rule_name", eff_name)
        object.__setattr__(self, "reject_code", eff_reject)
        object.__setattr__(self, "rejection_reason", eff_reject)
        object.__setattr__(self, "current_value", eff_cur)
        object.__setattr__(self, "actual_value", eff_cur)
        object.__setattr__(self, "limit_value", eff_lim)
        object.__setattr__(self, "threshold_value", eff_lim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "check_name": self.check_name,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "threshold_value": self.threshold_value,
            "rejection_reason": self.rejection_reason,
            "message": self.message,
        }


@dataclass(frozen=True)
class RiskDecision:
    """
    Kết quả tổng hợp sau khi đánh giá một OrderIntent qua toàn bộ các quy tắc của RiskManager.
    """

    status: RiskDecisionStatus
    approved_intent: OrderIntent | None = None
    reject_reason: str | None = None
    checks: tuple[RiskCheckResult, ...] = field(default_factory=tuple)
    sized_quantity: float | None = None
    estimated_risk_amount: float | None = None
    required_margin: float | None = None
    original_intent: OrderIntent | None = None
    estimated_risk: RiskEstimate | None = None
    audit_event_id: str = ""

    @property
    def is_approved(self) -> bool:
        return self.status == RiskDecisionStatus.APPROVED and self.approved_intent is not None

    @property
    def rejection_reason(self) -> str | None:
        return self.reject_reason

    @property
    def risk_checks(self) -> tuple[RiskCheckResult, ...]:
        return self.checks
