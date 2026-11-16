"""Các mô hình dữ liệu dùng chung của Group H."""
from __future__ import annotations

from enum import Enum


class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_RUN = "NOT_RUN"


_SEVERITY = {
    ValidationStatus.NOT_RUN: 0,
    ValidationStatus.PASS: 1,
    ValidationStatus.WARNING: 2,
    ValidationStatus.FAIL: 3,
}


def worst_status(*statuses: ValidationStatus) -> ValidationStatus:
    """Trả về trạng thái nghiêm trọng nhất (FAIL > WARNING > PASS > NOT_RUN)."""
    valid = [s for s in statuses if isinstance(s, ValidationStatus)]
    if not valid:
        return ValidationStatus.NOT_RUN
    return max(valid, key=lambda s: _SEVERITY[s])


class ValidationError(ValueError):
    """Lỗi đầu vào của các hàm Validation."""
