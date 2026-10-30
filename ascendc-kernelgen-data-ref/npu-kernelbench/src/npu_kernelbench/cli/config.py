"""Benchmark configuration model."""

from typing import Literal, Optional

from pydantic import field_validator, model_validator

from npu_kernelbench.data.base_model import BaseModelWithDocstrings


_ANTI_HACK_OPTIONS: frozenset[str] = frozenset(
    {"ast_scan", "kernel_signature", "output_caching", "monkey_patch", "clock_anomaly"}
)


class BenchmarkConfig(BaseModelWithDocstrings):
    """Configuration for benchmark evaluation."""

    warmup_runs: int = 5
    """GPU/NPU warmup iterations before timing."""
    iterations: int = 10
    """Timing iterations (median reported)."""
    seed: int = 200
    """RNG seed for input generation."""
    timeout: int = 600
    """Single problem timeout in seconds."""
    compile_timeout: int = 120
    """Compilation timeout in seconds."""
    benchmark_reference: bool = True
    """When True, also time the reference implementation."""
    npu_id: int = 0
    """(Deprecated) NPU device ID. Use npu_ids list instead."""
    profile_dir: str = ""
    """If set, save profiling traces to this directory."""
    runner: str = "persistent"
    """Runner mode: 'isolated' (process per eval) or 'persistent' (worker reuse)."""
    npu_ids: list[int] = [0]
    """NPU device IDs to use for evaluation."""
    work_dir: str = "/tmp/npu-kernelbench"
    """Working directory for staging build artifacts."""
    case_index: int = 0
    """0-based index of the current workload within the batch.

    Used to match legacy framework's per-case distribution alternation
    (e.g. idx % 2 selects normal vs uniform). Set by runners before each
    ``build_baseline`` call; user code normally leaves it at 0.
    """
    mode: Optional[str] = None
    """Evaluation mode. None = full benchmark (correctness + performance + anti-hack)
    ``"correctness_only"`` — skips performance profiling and anti-hack checks.
    This is the only valid non-None value; any other string raises a validation error."""
    enable_anti_hack: list[str] = ["ast_scan", "kernel_signature"]
    """Enabled anti-hack checks. Options: 'ast_scan', 'kernel_signature',
    'output_caching', 'monkey_patch', 'clock_anomaly'. Default: 'ast_scan' and
    'kernel_signature' enabled. Set to [] to disable all. Reward-hack checks
    ('output_caching', 'monkey_patch', 'clock_anomaly') must be enabled explicitly."""
    solution_source_dir: str = ""
    """Path to solution source directory. Required for AST scan and kernel signature.
    Set by runner before evaluation."""
    perf_reference_source: Literal["measured", "baseline"] = "measured"
    """Source for reference latency when computing speedup.
    ``"measured"`` (default) uses the timed reference implementation, falling
    back to ``workload.perf_baseline`` when the reference was not timed.
    ``"baseline"`` always uses ``workload.perf_baseline``.
    """

    @model_validator(mode="after")
    def _validate_config(self) -> "BenchmarkConfig":
        if self.warmup_runs < 0:
            raise ValueError(
                f"warmup_runs must be >= 0, got {self.warmup_runs}"
            )
        if self.iterations < 1:
            raise ValueError(
                f"iterations must be >= 1, got {self.iterations}"
            )
        if self.timeout < 1:
            raise ValueError(
                f"timeout must be >= 1, got {self.timeout}"
            )
        if self.compile_timeout < 1:
            raise ValueError(
                f"compile_timeout must be >= 1, got {self.compile_timeout}"
            )
        if self.runner not in ("isolated", "persistent"):
            raise ValueError(
                f"runner must be 'isolated' or 'persistent', got '{self.runner}'"
            )
        if not self.npu_ids:
            raise ValueError(
                f"npu_ids must be a non-empty list, got {self.npu_ids}"
            )
        return self

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "correctness_only":
            raise ValueError(
                f"mode must be correctness_only or None, got '{v}'"
            )
        return v

    @field_validator("perf_reference_source")
    @classmethod
    def _validate_perf_reference_source(cls, v: str) -> str:
        if v not in ("measured", "baseline"):
            raise ValueError(
                f"perf_reference_source must be 'measured' or 'baseline', got '{v}'"
            )
        return v

    @field_validator("enable_anti_hack")
    @classmethod
    def _validate_enable_anti_hack(cls, v: list[str]) -> list[str]:
        invalid = [x for x in v if x not in _ANTI_HACK_OPTIONS]
        if invalid:
            raise ValueError(
                f"Invalid anti-hack check(s): {invalid}. "
                f"Allowed: {sorted(_ANTI_HACK_OPTIONS)}"
            )
        return v
