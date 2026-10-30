"""NPU-KernelBench CLI — evaluate kernel solutions on Ascend NPU.

Usage:
    uv run npu-kernelbench eval <problem_dir> --solution solution.json
    uv run npu-kernelbench run-dataset <data_dir> --level level1
"""

from __future__ import annotations

# Disable torch's torch_npu autoloader in the parent CLI process. If torch_npu is
# auto-loaded here, ACL/GE initializes in the parent and later spawn subprocesses
# inherit a corrupted initialization state that causes AclSetCompileopt(500001)
# and "No module named 'tbe'" errors when convolution references run.
import os
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
import logging
from rich.logging import RichHandler

from npu_kernelbench.data import (
    Definition,
    Workload,
    Solution,
    Trace,
)
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.evaluator.generator import get_workload_mode
from npu_kernelbench.runner import IsolatedRunner, PersistentRunner, Runner
from npu_kernelbench.report import format_traces_table, compute_score, format_score_summary

console = Console(stderr=True)


class EvalError(Exception):
    """Raised when kernel evaluation fails in an expected way.

    Used instead of SystemExit so that _evaluate_single() is
    usable programmatically. The CLI layer catches this and
    translates to sys.exit(1).
    """


def _setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Configure logging with Rich console handler and optional file handler."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        RichHandler(console=console, show_time=True, show_path=False, rich_tracebacks=True)
    ]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        ))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def _load_definition(path: Path) -> Definition:
    return Definition(**json.loads(path.read_text()))


def _load_workloads(path: Path) -> list[Workload]:
    workloads = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            workloads.append(Workload(**json.loads(line)))
    return workloads


def _load_solution(path: Path) -> Solution:
    sol_dict = json.loads(path.read_text())
    sol_dir = path.parent
    for src in sol_dict.get("sources", []):
        if not src.get("content"):
            src_path = sol_dir / src["path"]
            if src_path.exists():
                src["content"] = src_path.read_text()
    return Solution(**sol_dict)


def _apply_limit(workloads: list[Workload], limit: int | None) -> list[Workload]:
    """Validate and apply workload limit.

    Raises click.ClickException if limit <= 0.
    Returns workloads truncated to min(limit, len(workloads)).
    """
    if limit is None:
        return workloads
    if limit <= 0:
        raise click.ClickException(f"--limit must be positive, got {limit}")
    return workloads[: min(limit, len(workloads))]


def _load_config(path: Optional[Path]) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig(**json.loads(path.read_text()))


def _resolve_problem_dir(
    problem_dir: Path,
) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    def_path = problem_dir / "definition.json"
    wkl_path = problem_dir / "workload.jsonl"
    cfg_path = problem_dir / "config.json"
    sol_path = problem_dir / "solution.json"
    if not def_path.exists():
        raise click.ClickException(f"definition.json not found in {problem_dir}")
    if not wkl_path.exists():
        raise click.ClickException(f"workload.jsonl not found in {problem_dir}")
    return (
        def_path,
        wkl_path,
        cfg_path if cfg_path.exists() else None,
        sol_path if sol_path.exists() else None,
    )


def _evaluate_with_runner(
    definition: Definition,
    workloads: list[Workload],
    solution: Solution,
    config: BenchmarkConfig,
    output_file: Path | None = None,
) -> list[Trace]:
    """Evaluate using the configured Runner.

    When output_file is provided, each trace is appended to the JSONL file
    immediately after the workload completes, so partial results survive a
    mid-run crash.
    """
    from npu_kernelbench.data.json_utils import append_jsonl_line

    runner_cls = {"isolated": IsolatedRunner, "persistent": PersistentRunner}[config.runner]
    runner_kwargs = {"npu_ids": config.npu_ids, "config": config}

    from npu_kernelbench.builder.builder import BuildError

    traces: list[Trace] = []
    try:
        with runner_cls(**runner_kwargs) as runner:
            runner.pre_build(definition, solution, config)
            traces = runner.run_workloads_batch(
                definition, workloads, solution,
            )
            if output_file:
                for trace in traces:
                    try:
                        append_jsonl_line(trace, output_file)
                    except Exception:
                        logging.warning(
                            "Failed to write incremental trace to %s", output_file,
                            exc_info=True,
                        )
    except BuildError as e:
        logging.error("Compilation failed for %s: %s", solution.name, e)
        traces = []
        for workload in workloads:
            trace = Runner._make_error_trace(
                definition, workload, solution, "COMPILE_ERROR", str(e),
            )
            traces.append(trace)
            if output_file:
                try:
                    append_jsonl_line(trace, output_file)
                except Exception:
                    logging.warning(
                        "Failed to write compile-error trace to %s", output_file,
                        exc_info=True,
                    )
    return traces


def _evaluate_single(
    definition: Definition,
    workloads: list[Workload],
    solution: Solution,
    config: BenchmarkConfig,
    output_file: Path | None = None,
) -> list[Trace]:
    """Run evaluation for a single problem and return traces."""
    if workloads:
        if config.mode == "correctness_only":
            mode = "correctness_only"
        else:
            mode = config.mode  # None → full benchmark
    else:
        mode = "correctness_only"
    logging.info("Problem:  %s", definition.name)
    logging.info("Solution: %s", solution.name)
    logging.info("Workloads: %s | Mode: %s", len(workloads), mode)

    traces = _evaluate_with_runner(
        definition, workloads, solution, config, output_file=output_file,
    )

    return traces


def _make_resume_key(
    definition_name: str, solution_name: Optional[str], uuid: str
) -> tuple[str, Optional[str], str]:
    """Identity key for a completed (definition, solution, workload) result."""
    return (definition_name, solution_name, uuid)


def _load_resume_index(path: Path) -> dict[tuple[str, Optional[str], str], Trace]:
    """Read the output JSONL and index traces that represent a completed result.

    Used by ``--resume`` to skip workloads/kernels already present in ``-o``.

    - File missing or empty -> returns ``{}`` (resume degrades to a fresh run).
    - Parses line by line; any line that fails ``Trace.model_validate``
      (including a half-written trailing line left by a killed process) is
      skipped with a warning. Resume never dies on one bad line.
    - Only traces whose ``evaluation`` is not None are indexed. Workload-only
      traces (no evaluation) are ignored for skipping but remain in the file.
    """
    index: dict[tuple[str, Optional[str], str], Trace] = {}
    if not path.exists():
        logging.info("Resume: output file %s not found; running fresh", path)
        return index

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                trace = Trace.model_validate_json(line)
            except Exception:
                logging.warning(
                    "Resume: skipping unparseable line %d in %s", lineno, path
                )
                continue
            if trace.evaluation is None:
                continue
            index[
                _make_resume_key(trace.definition, trace.solution, trace.workload.uuid)
            ] = trace
    return index


def _filter_workloads(
    workloads: list[Workload],
    definition_name: str,
    solution_name: Optional[str],
    index: dict[tuple[str, Optional[str], str], Trace],
) -> tuple[list[Workload], list[Trace]]:
    """Split ``workloads`` into ``(pending, reused)`` for ``--resume``.

    - ``pending``: workloads whose key is NOT in ``index`` (still need to run).
    - ``reused``: the existing ``Trace`` for each completed workload, in workload
      order. Empty ``pending`` means the kernel is fully done and the caller
      skips it entirely (no pre_build).
    """
    pending: list[Workload] = []
    reused: list[Trace] = []
    for wl in workloads:
        existing = index.get(
            _make_resume_key(definition_name, solution_name, wl.uuid)
        )
        if existing is not None:
            reused.append(existing)
        else:
            pending.append(wl)
    return pending, reused


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="npu-kernelbench")
def cli():
    """NPU KernelBench — Ascend NPU kernel evaluation framework."""


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "problem_dir", required=False, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--definition",
    "definition_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to definition.json",
)
@click.option(
    "--workload",
    "workload_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to workload.jsonl",
)
@click.option(
    "--solution",
    "solution_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to solution.json",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to benchmark config JSON",
)
@click.option(
    "--timeout", default=600, type=int, help="Evaluation subprocess timeout in seconds",
)
@click.option(
    "-o", "--output", "output_file", type=click.Path(path_type=Path),
    default=Path("/tmp/npu-kernelbench/traces.jsonl"),
    help="Write trace JSONL to this file",
)
@click.option(
    "--html", "html_report", type=click.Path(path_type=Path),
    help="Generate HTML report at the given path",
)
@click.option("--limit", type=int, default=None, help="Max workloads to evaluate")
@click.option("--json", "json_output", is_flag=True, help="Print trace JSON to stdout")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
@click.option(
    "--runner", default="persistent",
    type=click.Choice(["isolated", "persistent"]),
    help="Runner mode: 'persistent' (worker reuse, default) or 'isolated' (process per eval)",
)
@click.option(
    "--npu-ids", default="0",
    help="Comma-separated NPU device IDs (e.g., '0,1,2')",
)
@click.option(
    "--profile-dir", type=click.Path(path_type=Path),
    help="Save profiling traces to this directory",
)
@click.option(
    "--work-dir", type=click.Path(path_type=Path),
    default="/tmp/npu-kernelbench",
    help="Working directory for staging build artifacts",
)
@click.option(
    "--log-file", type=click.Path(path_type=Path),
    help="Write logs to this file",
)
@click.option(
    "--resume", is_flag=True,
    help="Resume from existing results in -o; skip workloads that already have a result",
)
@click.option(
    "--mode",
    type=click.Choice(["correctness_only"]),
    default=None,
    help="Run only correctness checks (skip performance and anti-hack). "
         "Default runs full benchmark.",
)
@click.option(
    "--perf-reference-source",
    type=click.Choice(["measured", "baseline"]),
    default="measured",
    help="Source for reference latency when computing speedup. "
         "'measured' uses timed reference (default); 'baseline' uses workload.perf_baseline.",
)
@click.option(
    "--enable-anti-hack", "enable_anti_hack_opts",
    multiple=True, default=None,
    type=click.Choice([
        "ast_scan", "kernel_signature", "output_caching",
        "monkey_patch", "clock_anomaly",
    ]),
    help="Enable a specific anti-hack check (repeatable). "
         "Default: ast_scan and kernel_signature. To disable all, set "
         "enable_anti_hack: [] in config.json.",
)
def eval(
    problem_dir: Optional[Path],
    definition_file: Optional[Path],
    workload_file: Optional[Path],
    solution_file: Optional[Path],
    config_file: Optional[Path],
    timeout: int,
    limit: Optional[int],
    output_file: Path,
    html_report: Optional[Path],
    json_output: bool,
    verbose: bool,
    runner: str,
    npu_ids: str,
    profile_dir: Optional[Path],
    work_dir: Optional[Path],
    log_file: Optional[Path],
    resume: bool = False,
    enable_anti_hack_opts: tuple[str, ...] | None = None,
    mode: Optional[str] = None,
    perf_reference_source: str = "measured",
):
    """Evaluate a single kernel solution.

    \b
    Two ways to specify the problem:
      1) Positional: npu-kernelbench eval <problem_dir> --solution sol.json
      2) Explicit:   npu-kernelbench eval --definition def.json --workload wkl.jsonl --solution sol.json
    """
    if problem_dir:
        def_path, wkl_path, cfg_path, sol_path = _resolve_problem_dir(problem_dir)
        definition_file = definition_file or def_path
        workload_file = workload_file or wkl_path
        config_file = config_file or cfg_path
        solution_file = solution_file or sol_path

    if not definition_file:
        raise click.ClickException("Provide PROBLEM_DIR or --definition")
    if not workload_file:
        raise click.ClickException("Provide PROBLEM_DIR or --workload")
    if not solution_file:
        raise click.ClickException("Provide PROBLEM_DIR with solution.json or --solution")

    definition = _load_definition(definition_file)
    workloads = _apply_limit(_load_workloads(workload_file), limit)
    solution = _load_solution(solution_file)
    config = _load_config(config_file)
    config.timeout = timeout
    config.runner = runner
    config.npu_ids = [int(x.strip()) for x in npu_ids.split(",") if x.strip()]
    config.mode = mode
    config.perf_reference_source = perf_reference_source
    if profile_dir:
        config.profile_dir = str(profile_dir)
    if work_dir:
        config.work_dir = str(work_dir)
    if enable_anti_hack_opts:
        config.enable_anti_hack = list(enable_anti_hack_opts)

    _setup_logging(verbose=verbose, log_file=str(log_file) if log_file else None)

    reused: list[Trace] = []
    pending_workloads = workloads
    if resume:
        index = _load_resume_index(output_file)
        pending_workloads, reused = _filter_workloads(
            workloads, definition.name, solution.name, index
        )
        logging.info(
            "Resume: %d/%d workload(s) already done, %d to run",
            len(reused), len(workloads), len(pending_workloads),
        )

    traces: list[Trace] = []
    try:
        if resume and not pending_workloads:
            logging.info("Resume: all workloads already done — skipping build/eval")
        else:
            traces = _evaluate_single(
                definition, pending_workloads if resume else workloads,
                solution, config, output_file=output_file,
            )
        final = reused + traces

        # Output
        if json_output:
            for t in final:
                console.print(t.model_dump_json(indent=2))
        else:
            format_traces_table(final)
            score = compute_score(final)
            console.print(f"\n[bold]Score Summary:[/bold]\n{format_score_summary(score)}")

        if output_file:
            if resume:
                logging.info(
                    "Resume: preserved existing + appended new traces in %s",
                    output_file,
                )
            else:
                from npu_kernelbench.data import save_jsonl_file
                save_jsonl_file(final, output_file)
                logging.info("Traces written to %s", output_file)

        if html_report:
            from npu_kernelbench.report.html_reporter import HtmlReporter
            HtmlReporter().write_report(final, html_report)
            logging.info("HTML report written to %s", html_report)
    except EvalError as e:
        logging.error("%s", e)
        raise SystemExit(1) from e
    except Exception as e:
        logging.error("Unexpected error: %s", e, exc_info=True)
        # On resume, never overwrite — incremental appends already captured
        # completed workloads and old traces must be preserved (union).
        if output_file and traces and not resume:
            try:
                from npu_kernelbench.data import save_jsonl_file
                save_jsonl_file(traces, output_file)
                logging.info("Partial traces saved to %s", output_file)
            except Exception:
                logging.warning("Failed to save partial traces", exc_info=True)
        raise SystemExit(1) from e


@cli.command(name="run-dataset", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "data_dir", required=False, default=Path("data"),
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--scenario",
    type=click.Choice(["ascendc_migration", "kernel_generator", "triton_migration"]),
    help="Benchmark scenario under data_dir (e.g., ascendc_migration).",
)
@click.option(
    "--level", "-l", multiple=True, default=["level1"],
    help="Levels to evaluate (default: level1, repeatable)",
)
@click.option("--limit", type=int, default=None, help="Max workloads per kernel")
@click.option(
    "--solution-dir", type=click.Path(exists=True, path_type=Path),
    help="Directory containing solution files per kernel",
)
@click.option("--verbose", "-v", is_flag=True, help="Show subprocess output")
@click.option(
    "--timeout", default=600, type=int, help="Evaluation subprocess timeout in seconds",
)
@click.option(
    "--runner", default="persistent",
    type=click.Choice(["isolated", "persistent"]),
    help="Runner mode: 'persistent' (default) or 'isolated'",
)
@click.option(
    "--npu-ids", default="0",
    help="Comma-separated NPU device IDs (e.g., '0,1,2')",
)
@click.option(
    "--profile-dir", type=click.Path(path_type=Path),
    help="Save profiling traces to this directory",
)
@click.option(
    "--work-dir", type=click.Path(path_type=Path),
    default="/tmp/npu-kernelbench",
    help="Working directory for staging build artifacts",
)
@click.option(
    "--log-file", type=click.Path(path_type=Path),
    help="Write logs to this file",
)
@click.option(
    "-o", "--output", "output_file", type=click.Path(path_type=Path),
    default=Path("/tmp/npu-kernelbench/traces.jsonl"),
    help="Write trace JSONL to this file",
)
@click.option(
    "--html", "html_report", type=click.Path(path_type=Path),
    help="Generate aggregated HTML report at the given path",
)
@click.option(
    "--resume", is_flag=True,
    help="Resume from existing results in -o; skip workloads/kernels that already have a result",
)
@click.option(
    "--mode",
    type=click.Choice(["correctness_only"]),
    default=None,
    help="Force evaluation mode: 'correctness_only' skips performance profiling "
         "and anti-hack checks. Default performs full benchmark.",
)
@click.option(
    "--perf-reference-source",
    type=click.Choice(["measured", "baseline"]),
    default="measured",
    help="Source for reference latency when computing speedup. "
         "'measured' uses timed reference (default); 'baseline' uses workload.perf_baseline.",
)
@click.option(
    "--enable-anti-hack", "enable_anti_hack_opts",
    multiple=True, default=None,
    type=click.Choice([
        "ast_scan", "kernel_signature", "output_caching",
        "monkey_patch", "clock_anomaly",
    ]),
    help="Enable a specific anti-hack check (repeatable). "
         "Default: ast_scan and kernel_signature. "
         "To disable all, set enable_anti_hack: [] in config.json.",
)
def run_dataset(
    data_dir: Path,
    scenario: Optional[str],
    level: list[str],
    limit: Optional[int],
    solution_dir: Optional[Path],
    verbose: bool,
    timeout: int,
    runner: str,
    npu_ids: str,
    profile_dir: Optional[Path],
    work_dir: Optional[Path],
    log_file: Optional[Path],
    output_file: Path,
    html_report: Optional[Path] = None,
    resume: bool = False,
    mode: Optional[str] = None,
    perf_reference_source: str = "measured",
    enable_anti_hack_opts: tuple[str, ...] | None = None,
):
    """Batch evaluate all kernels in a dataset directory.

    Expects directory structure:
        data_dir/scenario/level1/KernelName/definition.json + workload.jsonl

    When --scenario is given, data_dir is resolved to data_dir/scenario.
    """
    if scenario:
        data_dir = data_dir / scenario
        if not data_dir.exists():
            raise click.ClickException(f"Scenario directory not found: {data_dir}")
    _setup_logging(verbose=verbose, log_file=str(log_file) if log_file else None)

    resume_index = _load_resume_index(output_file) if resume else {}
    if resume:
        logging.info(
            "Resume: %s completed result(s) loaded from %s",
            len(resume_index), output_file,
        )

    total_kernels = 0
    total_workloads = 0
    all_scores = []
    all_traces: list[Trace] = []

    for lvl in level:
        level_dir = data_dir / lvl
        if not level_dir.exists():
            logging.warning("Skipping %s: not found", lvl)
            continue

        kernel_dirs = sorted(
            d for d in level_dir.iterdir() if d.is_dir()
        )
        if not kernel_dirs:
            logging.warning("No kernels in %s", lvl)
            continue

        for kd in kernel_dirs:
            def_path = kd / "definition.json"
            wl_path = kd / "workload.jsonl"
            sol_path = (
                solution_dir / lvl / kd.name / "solution.json"
                if solution_dir else kd / "solution.json"
            )

            if not def_path.exists():
                logging.warning("Skip %s: no definition.json", kd.name)
                continue
            if not wl_path.exists():
                logging.warning("Skip %s: no workload.jsonl", kd.name)
                continue

            logging.info("=== %s/%s ===", lvl, kd.name)

            try:
                definition = _load_definition(def_path)
                workloads = _apply_limit(_load_workloads(wl_path), limit)
                solution = _load_solution(sol_path) if sol_path.exists() else None
                config = BenchmarkConfig(
                    timeout=timeout,
                    runner=runner,
                    npu_ids=[int(x.strip()) for x in npu_ids.split(",") if x.strip()],
                    mode=mode,
                    perf_reference_source=perf_reference_source,
                )
                if enable_anti_hack_opts:
                    config.enable_anti_hack = list(enable_anti_hack_opts)
                if profile_dir:
                    config.profile_dir = str(profile_dir)
                if work_dir:
                    config.work_dir = str(work_dir)

                if solution is None:
                    logging.warning("Skip %s: no solution", kd.name)
                    continue

                total_kernels += 1
                total_workloads += len(workloads)

                reused: list[Trace] = []
                pending_workloads = workloads
                if resume:
                    pending_workloads, reused = _filter_workloads(
                        workloads, definition.name, solution.name, resume_index
                    )

                if resume and not pending_workloads:
                    logging.info(
                        "Skip %s/%s: all %d workload(s) already done",
                        lvl, kd.name, len(reused),
                    )
                    kernel_traces = reused  # nothing new to run; no pre_build
                else:
                    if resume:
                        logging.info(
                            "Resume %s/%s: %d done, %d to run",
                            lvl, kd.name, len(reused), len(pending_workloads),
                        )
                    new_traces = _evaluate_single(
                        definition, pending_workloads, solution, config,
                        output_file=output_file,
                    )
                    kernel_traces = reused + new_traces

                all_scores.append(compute_score(kernel_traces))
                all_traces.extend(kernel_traces)

            except EvalError as e:
                logging.error("Eval error for %s: %s", kd.name, e)
                continue
            except Exception as e:
                logging.error("Unexpected error for %s: %s", kd.name, e,
                              exc_info=True)
                continue

    # Final aggregate
    logging.info("Done. %s kernels, %s workloads", total_kernels, total_workloads)
    if all_scores:
        total_passed = sum(s.passed for s in all_scores)
        total_all = sum(s.total for s in all_scores)
        logging.info("Overall pass rate: %s/%s", total_passed, total_all)

    if all_traces:
        if resume:
            # New traces were appended incrementally; existing traces preserved.
            logging.info(
                "Resume: preserved existing + appended new traces in %s",
                output_file,
            )
        else:
            from npu_kernelbench.data import save_jsonl_file
            save_jsonl_file(all_traces, output_file)
            logging.info("Traces written to %s", output_file)

    if html_report and all_traces:
        from npu_kernelbench.report.html_reporter import HtmlReporter
        HtmlReporter().write_report(all_traces, html_report)
        logging.info("HTML report written to %s", html_report)


if __name__ == "__main__":
    cli()
