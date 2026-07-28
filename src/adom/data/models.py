from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class DatasetError(RuntimeError):
    """Raised when a dataset violates the canonical ADOM contract."""


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    sequence: str
    image_path: Path
    source_mask_path: Path
    split: str | None = None


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    detail: str
    sample_id: str = ""


@dataclass
class ValidationReport:
    dataset: str
    version: str = ""
    issues: list[ValidationIssue] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def error(self, code: str, detail: str, sample_id: str = "") -> None:
        self.issues.append(ValidationIssue("error", code, detail, sample_id))

    def warning(self, code: str, detail: str, sample_id: str = "") -> None:
        self.issues.append(ValidationIssue("warning", code, detail, sample_id))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "status": "PASS" if self.passed else "FAIL",
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(issue) for issue in self.issues],
            "statistics": self.statistics,
        }

    def require_success(self) -> None:
        if self.errors:
            examples = "; ".join(
                f"{item.code}:{item.sample_id or '-'}:{item.detail}"
                for item in self.errors[:5]
            )
            raise DatasetError(
                f"{self.dataset} validation failed with "
                f"{len(self.errors)} error(s): {examples}"
            )
