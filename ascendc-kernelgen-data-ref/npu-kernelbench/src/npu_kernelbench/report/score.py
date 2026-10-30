"""NPU scoring — compute performance score from evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass, field

from npu_kernelbench.data.trace import Trace, EvaluationStatus


@dataclass
class ScoreSummary:
    """Aggregated score summary from a set of traces."""

    total: int = 0
    passed: int = 0
    runtime_errors: int = 0
    compile_errors: int = 0
    incorrect_numerical: int = 0
    incorrect_shape: int = 0
    incorrect_dtype: int = 0
    invalid_reference: int = 0
    timeout: int = 0
    reward_hacks: int = 0
    crashes: int = 0

    pass_rate: float = 0.0
    avg_speedup: float | None = None
    max_speedup: float | None = None
    min_speedup: float | None = None
    median_speedup: float | None = None

    avg_latency_ms: float | None = None
    median_latency_ms: float | None = None

    extra: dict = field(default_factory=dict)


def compute_score(traces: list[Trace]) -> ScoreSummary:
    """Compute score summary from evaluation traces."""
    s = ScoreSummary(total=len(traces))

    speedups = []
    latencies = []

    for trace in traces:
        ev = trace.evaluation
        if ev is None:
            continue

        status = ev.status
        if status == EvaluationStatus.PASSED:
            s.passed += 1
            if ev.performance:
                latencies.append(ev.performance.latency_ms)
                if ev.performance.speedup_factor > 0:
                    speedups.append(ev.performance.speedup_factor)
        elif status == EvaluationStatus.RUNTIME_ERROR:
            s.runtime_errors += 1
        elif status == EvaluationStatus.COMPILE_ERROR:
            s.compile_errors += 1
        elif status == EvaluationStatus.INCORRECT_NUMERICAL:
            s.incorrect_numerical += 1
        elif status == EvaluationStatus.INCORRECT_SHAPE:
            s.incorrect_shape += 1
        elif status == EvaluationStatus.INCORRECT_DTYPE:
            s.incorrect_dtype += 1
        elif status == EvaluationStatus.INVALID_REFERENCE:
            s.invalid_reference += 1
        elif status == EvaluationStatus.TIMEOUT:
            s.timeout += 1
        elif status == EvaluationStatus.CRASH:
            s.crashes += 1
        elif status == EvaluationStatus.REWARD_HACK:
            s.reward_hacks += 1

    if s.total > 0:
        s.pass_rate = s.passed / s.total

    if speedups:
        import statistics
        s.avg_speedup = statistics.mean(speedups)
        s.max_speedup = max(speedups)
        s.min_speedup = min(speedups)
        s.median_speedup = statistics.median(speedups)

    if latencies:
        import statistics
        s.avg_latency_ms = statistics.mean(latencies)
        s.median_latency_ms = statistics.median(latencies)

    return s


def format_score_summary(s: ScoreSummary) -> str:
    """Format a ScoreSummary as a human-readable string."""
    lines = [
        f"Total: {s.total}",
        f"Passed: {s.passed} ({s.pass_rate:.1%})",
    ]
    if s.runtime_errors:
        lines.append(f"Runtime Errors: {s.runtime_errors}")
    if s.compile_errors:
        lines.append(f"Compile Errors: {s.compile_errors}")
    if s.incorrect_numerical:
        lines.append(f"Incorrect Numerical: {s.incorrect_numerical}")
    if s.incorrect_shape:
        lines.append(f"Incorrect Shape: {s.incorrect_shape}")
    if s.incorrect_dtype:
        lines.append(f"Incorrect Dtype: {s.incorrect_dtype}")
    if s.invalid_reference:
        lines.append(f"Invalid Reference: {s.invalid_reference}")
    if s.timeout:
        lines.append(f"Timeout: {s.timeout}")
    if s.crashes:
        lines.append(f"Crashes: {s.crashes}")
    if s.reward_hacks:
        lines.append(f"Reward Hacks: {s.reward_hacks}")
    if s.avg_speedup is not None:
        lines.append(f"Avg Speedup: {s.avg_speedup:.2f}x")
    if s.max_speedup is not None:
        lines.append(f"Max Speedup: {s.max_speedup:.2f}x")
    if s.median_latency_ms is not None:
        lines.append(f"Median Latency: {s.median_latency_ms:.3f} ms")
    return "\n".join(lines)
