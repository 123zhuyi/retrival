"""Runner ABC — abstract interface for evaluating solution-workload pairs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.data.trace import Trace, Evaluation, EvaluationStatus, Environment


def _partition(workloads: list[Workload], num_buckets: int) -> list[list[Workload]]:
    """Distribute workloads evenly across buckets (round-robin)."""
    buckets: list[list[Workload]] = [[] for _ in range(num_buckets)]
    for i, wl in enumerate(workloads):
        buckets[i % num_buckets].append(wl)
    return buckets


class Runner(ABC):
    """Abstract runner for evaluating solution-workload pairs."""

    def __init__(
        self,
        npu_ids: list[int] | None = None,
        config: BenchmarkConfig | None = None,
    ):
        from npu_kernelbench.cli.config import BenchmarkConfig as _BC

        self._npu_ids = npu_ids or [0]
        self._config = config or _BC()
        self._work_dir: Path | None = None

    # ── concrete: build pipeline ────────────────────────────────────────

    def pre_build(
        self,
        definition: Definition,
        solution: Solution,
        config: BenchmarkConfig,
    ) -> Path:
        """Create work_dir, stage sources, compile, load .so, return path.

        Called once before the workload loop. Subclasses inherit this
        directly — they only need to override if custom staging is required.
        """
        work_dir = (
            Path(config.work_dir)
            / f"{definition.name}-{solution.name}-{solution.author}"
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        sources_dir = work_dir / "sources"
        sources_dir.mkdir(exist_ok=True)
        for src in solution.sources:
            (sources_dir / src.path).parent.mkdir(parents=True, exist_ok=True)
            (sources_dir / src.path).write_text(src.content)

        self._stage_eval_files(work_dir, definition, solution)

        from npu_kernelbench.builder.registry import get_registry as get_builder

        # build() is invoked for its side effect: it compiles kernel.so into
        # work_dir. It returns nothing; the kernel fn is loaded per-worker.
        get_builder().build(definition, solution, work_dir=str(work_dir),
                            compile_timeout=config.compile_timeout)

        so_files = list(work_dir.glob("*.so"))
        if so_files:
            import torch

            torch.ops.load_library(str(so_files[0]))

        (work_dir / "profile").mkdir(exist_ok=True)
        self._work_dir = work_dir
        return work_dir

    # ── static helpers ──────────────────────────────────────────────────

    @staticmethod
    def _stage_eval_files(
        work_dir: Path,
        definition: Definition,
        solution: Solution,
    ) -> None:
        """Stage definition / solution for standalone reproduction."""
        (work_dir / "definition.json").write_text(
            definition.model_dump_json(indent=2)
        )
        (work_dir / "solution.json").write_text(
            solution.model_dump_json(indent=2)
        )

    @staticmethod
    def _append_workload(work_dir: Path, workload: Workload) -> None:
        """Append a single workload line to workload.jsonl in *work_dir*."""
        wl_path = work_dir / "workload.jsonl"
        line = workload.model_dump_json(indent=None) + "\n"
        with open(wl_path, "a") as f:
            f.write(line)

    @staticmethod
    def _make_error_trace(
        definition: Definition,
        workload: Workload,
        solution: Solution,
        status: str,
        log: str,
    ) -> Trace:
        """Build a Trace for a non-passing evaluation (error / timeout / …)."""
        import os

        import torch

        try:
            if torch.npu.is_available():
                hw = torch.npu.get_device_name() or "Ascend_Unknown"
            else:
                hw = os.environ.get("NPU_HARDWARE_NAME", "Ascend_Unknown")
        except Exception:
            hw = os.environ.get("NPU_HARDWARE_NAME", "Ascend_Unknown")

        # This is the error path — it must never raise, or the secondary
        # exception masks the real worker failure. An unknown status string
        # (typo, new code path) degrades to RUNTIME_ERROR rather than crashing.
        try:
            status_enum = EvaluationStatus(status)
        except ValueError:
            status_enum = EvaluationStatus.RUNTIME_ERROR

        return Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=status_enum,
                environment=Environment(hardware=hw),
                timestamp=datetime.now(timezone.utc).isoformat(),
                log=log,
            ),
        )

    @staticmethod
    def _load_solution_fn(
        work_dir: str | Path,
        solution: Solution,
    ) -> Callable:
        """Load .so and import solution entry point, return callable."""
        import importlib.util
        import sys

        import torch

        for so_file in Path(work_dir).glob("*.so"):
            torch.ops.load_library(str(so_file))

        entry_file = solution.get_entry_path()
        entry_symbol = solution.get_entry_symbol()
        sources_dir = Path(work_dir) / "sources"
        module_path = sources_dir / entry_file

        spec = importlib.util.spec_from_file_location(
            "npu_kernelbench_user_solution", str(module_path)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["npu_kernelbench_user_solution"] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, entry_symbol)

    # ── abstract interface ──────────────────────────────────────────────

    @abstractmethod
    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solution: Solution,
        case_index: int = 0,
    ) -> Trace:
        """Evaluate one solution against one workload. Return a Trace.

        ``case_index`` is the 0-based index of this workload within the
        evaluation batch. It is forwarded to input generators so that
        legacy per-case distribution alternation (e.g. idx % 2) is
        reproducible across single-workload and batch invocations.
        """
        ...

    @abstractmethod
    def run_workloads_batch(
        self,
        definition: Definition,
        workloads: list[Workload],
        solution: Solution,
    ) -> list[Trace]:
        """Evaluate one solution against multiple workloads in parallel.

        Dispatches workloads across available NPU devices concurrently.
        Returns traces in the same order as the input workloads list.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Shutdown all workers and release resources."""
        ...

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
