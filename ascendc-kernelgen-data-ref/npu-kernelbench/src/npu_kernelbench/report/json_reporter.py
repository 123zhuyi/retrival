"""JsonReporter — serializes evaluation traces to JSON/JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from npu_kernelbench.data.trace import Trace
from npu_kernelbench.report.score import compute_score


class JsonReporter:
    """Serialize evaluation traces to JSON or JSONL format.

    Usage:
        reporter = JsonReporter(indent=2)
        reporter.write_report(traces, Path("report.json"))
        # Or stream:
        reporter.append_trace(trace, Path("traces.jsonl"))
    """

    def __init__(self, indent: int | None = 2):
        self._indent = indent

    def write_report(
        self,
        traces: list[Trace],
        path: Path,
        *,
        include_score: bool = True,
    ) -> None:
        """Write all traces as a JSON array to path.

        When include_score is True, embeds a score_summary key.
        """
        data: dict = {
            "traces": [t.model_dump(mode="json") for t in traces],
        }
        if include_score:
            score = compute_score(traces)
            data["score_summary"] = {
                "total": score.total,
                "passed": score.passed,
                "pass_rate": score.pass_rate,
                "runtime_errors": score.runtime_errors,
                "compile_errors": score.compile_errors,
                "incorrect_numerical": score.incorrect_numerical,
                "incorrect_shape": score.incorrect_shape,
                "incorrect_dtype": score.incorrect_dtype,
                "invalid_reference": score.invalid_reference,
                "timeout": score.timeout,
                "reward_hacks": score.reward_hacks,
                "avg_speedup": score.avg_speedup,
                "max_speedup": score.max_speedup,
                "median_speedup": score.median_speedup,
                "avg_latency_ms": score.avg_latency_ms,
                "median_latency_ms": score.median_latency_ms,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=self._indent, ensure_ascii=False))

    def append_trace(self, trace: Trace, path: Path) -> None:
        """Append a single trace as a JSON line to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = trace.model_dump_json(indent=None)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
