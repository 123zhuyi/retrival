"""Trace and Evaluation models for benchmark results."""

from enum import Enum
from typing import Any, Optional

from pydantic import Field, field_validator, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString
from .workload import Workload


class Correctness(BaseModelWithDocstrings):
    """Correctness metrics from numerical evaluation."""

    max_relative_error: float = Field(default=0.0, ge=0.0)
    """Maximum relative error across all output elements."""
    max_absolute_error: float = Field(default=0.0, ge=0.0)
    """Maximum absolute error across all output elements."""
    has_nan: bool = Field(default=False)
    """True when output contains NaN values."""
    has_inf: bool = Field(default=False)
    """True when output contains Inf values (but no NaN)."""
    extra: Optional[dict[str, Any]] = Field(default=None)
    """Extra metrics for correctness evaluation."""

    @field_validator("max_relative_error", "max_absolute_error")
    @classmethod
    def non_negative(cls, v: float):
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class Performance(BaseModelWithDocstrings):
    """Performance metrics from timing evaluation."""

    latency_ms: float = Field(default=0.0, ge=0.0)
    """Solution execution latency in milliseconds."""
    reference_latency_ms: float = Field(default=0.0, ge=0.0)
    """Reference implementation latency in milliseconds."""
    speedup_factor: float = Field(default=0.0, ge=0.0)
    """Performance speedup (reference_time / solution_time)."""


class Environment(BaseModelWithDocstrings):
    """Environment information from evaluation execution."""

    hardware: NonEmptyString
    """Hardware identifier (e.g., 'Ascend_910B')."""
    libs: dict[str, str] = Field(default_factory=dict)
    """Library name to version mapping."""


class EvaluationStatus(str, Enum):
    """Status codes for evaluation results."""

    PASSED = "PASSED"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILE_ERROR = "COMPILE_ERROR"
    TIMEOUT = "TIMEOUT"
    CRASH = "CRASH"
    REWARD_HACK = "REWARD_HACK"


class Evaluation(BaseModelWithDocstrings):
    """Complete evaluation result for a solution on a workload."""

    status: EvaluationStatus
    """Overall evaluation status."""
    environment: Environment
    """Environment details."""
    timestamp: NonEmptyString
    """Timestamp in ISO format."""
    log: str = ""
    """Captured stdout/stderr from the evaluation run."""
    correctness: Optional[Correctness] = None
    """Correctness metrics (PASSED and INCORRECT_NUMERICAL only)."""
    performance: Optional[Performance] = None
    """Performance metrics (PASSED only)."""

    @model_validator(mode="after")
    def _validate_status_correctness_performance(self) -> "Evaluation":
        if self.status == EvaluationStatus.PASSED:
            if self.correctness is None:
                raise ValueError(
                    f"Must include correctness when status is {self.status}"
                )
            if self.performance is None:
                raise ValueError(
                    f"Must include performance when status is {self.status}"
                )
        elif self.status == EvaluationStatus.INCORRECT_NUMERICAL:
            if self.correctness is None:
                raise ValueError(
                    f"Must include correctness when status is {self.status}"
                )
            if self.performance is not None:
                raise ValueError(
                    f"Must not include performance when status is {self.status}"
                )
        else:
            if self.correctness is not None:
                raise ValueError(
                    f"Must not include correctness when status is {self.status}"
                )
            if self.performance is not None:
                raise ValueError(
                    f"Must not include performance when status is {self.status}"
                )
        return self


class Trace(BaseModelWithDocstrings):
    """Complete trace linking a solution to a definition with evaluation results."""

    definition: NonEmptyString
    """Name of the Definition."""
    workload: Workload
    """Concrete workload configuration."""
    solution: Optional[str] = None
    """Name of the Solution (None for workload-only traces)."""
    evaluation: Optional[Evaluation] = None
    """Evaluation results (None for workload-only traces)."""

    def is_workload_trace(self) -> bool:
        return self.solution is None and self.evaluation is None

    def is_successful(self) -> bool:
        return (
            not self.is_workload_trace()
            and self.evaluation is not None
            and self.evaluation.status == EvaluationStatus.PASSED
        )
