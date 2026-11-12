from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IssueSeverity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    severity: IssueSeverity
    code: str
    message: str
    row_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    infos: list[ValidationIssue] = field(default_factory=list)
    total_rows: int = 0
    invalid_rows: int = 0

    def add_error(self, code: str, message: str, row_indices: list[int] | None = None, **meta):
        self.errors.append(
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                code=code,
                message=message,
                row_indices=row_indices or [],
                metadata=meta,
            )
        )
        self.is_valid = False

    def add_warning(self, code: str, message: str, row_indices: list[int] | None = None, **meta):
        self.warnings.append(
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                code=code,
                message=message,
                row_indices=row_indices or [],
                metadata=meta,
            )
        )

    def add_info(self, code: str, message: str, **meta):
        self.infos.append(
            ValidationIssue(
                severity=IssueSeverity.INFO,
                code=code,
                message=message,
                row_indices=[],
                metadata=meta,
            )
        )

