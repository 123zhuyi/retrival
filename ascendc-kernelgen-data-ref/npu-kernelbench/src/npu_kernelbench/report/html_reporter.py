"""HtmlReporter — generates standalone HTML evaluation reports via Jinja2."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import jinja2

from npu_kernelbench.data.trace import Trace
from npu_kernelbench.report.score import compute_score

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    undefined=jinja2.StrictUndefined,
    autoescape=False,
)


def _render_template(name: str, **context: object) -> str:
    """Load and render a Jinja2 template from the templates/ directory."""
    template = _jinja_env.get_template(name)
    return template.render(**context)


class HtmlReporter:
    """Generate a standalone HTML report page from evaluation traces.

    Zero external dependencies — inline CSS/JS, single file, open in any browser.
    Traces are grouped by solution name with per-solution stat cards and tables.

    Usage:
        reporter = HtmlReporter()
        reporter.write_report(traces, Path("report.html"))
    """

    def write_report(
        self,
        traces: list[Trace],
        path: Path,
        *,
        title: str = "NPU KernelBench Report",
    ) -> None:
        html = self._build_html(traces, title)
        path.write_text(html, encoding="utf-8")

    def _build_html(self, traces: list[Trace], title: str) -> str:
        solutions = self._group_traces(traces)
        return _render_template("report.html.j2",
            title=title,
            solutions=solutions,
        )

    @staticmethod
    def _group_traces(traces: list[Trace]) -> list[dict]:
        """Group traces by solution, compute per-group score and row data."""
        groups: dict[str, list[Trace]] = OrderedDict()
        for t in traces:
            name = t.solution or "(no solution)"
            groups.setdefault(name, []).append(t)

        solutions = []
        for name, group_traces in groups.items():
            score = compute_score(group_traces)
            rows = HtmlReporter._extract_rows(group_traces)
            definition = group_traces[0].definition
            solutions.append({
                "name": name,
                "definition": definition,
                "score": score,
                "rows": rows,
            })
        return solutions

    @staticmethod
    def _extract_rows(traces: list[Trace]) -> list[dict]:
        """Extract table row data from traces (no HTML)."""
        rows: list[dict] = []
        for trace in traces:
            ev = trace.evaluation
            workload_uuid = trace.workload.uuid
            hardware = ev.environment.hardware if ev else ""

            if ev is None:
                rows.append({
                    "uuid": workload_uuid, "hardware": hardware,
                    "status": "", "latency": "",
                    "ref_latency": "", "speedup": "",
                    "abs_err": "", "rel_err": "", "log": "",
                })
                continue

            status = ev.status.value
            latency = f"{ev.performance.latency_ms:.3f}" if ev.performance else ""
            ref_lat = f"{ev.performance.reference_latency_ms:.3f}" if ev.performance else ""
            speedup = f"{ev.performance.speedup_factor:.2f}x" if ev.performance else ""

            abs_err = ""
            rel_err = ""
            if ev.correctness:
                if ev.correctness.has_nan:
                    abs_err = "NaN"
                    rel_err = "NaN"
                elif ev.correctness.has_inf:
                    abs_err = "Inf"
                    rel_err = "Inf"
                else:
                    abs_err = f"{ev.correctness.max_absolute_error:.2e}"
                    rel_err = f"{ev.correctness.max_relative_error:.2e}"

            rows.append({
                "uuid": workload_uuid,
                "hardware": hardware,
                "status": status,
                "latency": latency,
                "ref_latency": ref_lat,
                "speedup": speedup,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "log": (ev.log or "")[:120],
            })
        return rows
