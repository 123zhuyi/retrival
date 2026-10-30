"""IsolatedRunner — process-per-evaluation runner with full NPU isolation."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from pathlib import Path

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.data.trace import Trace
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.runner.runner import Runner, _partition


def _isolated_batch_worker(
    conn: mp.connection.Connection,
    config: BenchmarkConfig,
    definition: Definition,
    tasks: list[tuple[Workload, int, int]],
    solution: Solution,
    work_dir: str,
) -> None:
    """Process a list of (workload, global_index, npu_id) tasks, send all traces back."""
    import os

    # Prevent torch from auto-loading torch_npu before we set ASCEND_RT_VISIBLE_DEVICES.
    # Otherwise ACL/GE initializes against the wrong device state and later
    # AclSetCompileopt calls fail with 500001 in this subprocess.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    if tasks:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(tasks[0][2])
    import torch
    import torch_npu  # noqa: F401
    # Pre-import tbe so that reference code setting torch_npu.npu.conv.allow_hf32
    # can initialise the CANN compiler in this subprocess.
    try:
        import tbe  # noqa: F401
    except Exception:
        pass
    device = "npu:0"

    traces: list[Trace] = []
    for workload, case_index, _npu_id in tasks:
        try:
            # 1. Load .so and import solution entry point
            solution_fn = Runner._load_solution_fn(work_dir, solution)

            # 2. Point AST scan / kernel signature at staged sources
            config.solution_source_dir = str(Path(work_dir) / "sources")

            # 3. Build baseline
            from npu_kernelbench.evaluator.registry import get_registry as get_eval

            evaluator = get_eval().resolve(definition)
            config.case_index = case_index
            config.profile_dir = str(
                Path(work_dir) / "profile" / f"{workload.uuid}_ref"
            )
            baseline = evaluator.build_baseline(
                definition, workload, device, config
            )

            # 4. Evaluate
            config.profile_dir = str(
                Path(work_dir) / "profile" / f"{workload.uuid}_sol"
            )
            trace = evaluator.evaluate(
                definition, workload, solution_fn, baseline, config, solution=solution
            )
            trace.solution = solution.name
        except Exception as e:
            trace = Runner._make_error_trace(
                definition, workload, solution, "RUNTIME_ERROR",
                f"{e}\n{traceback.format_exc()}",
            )
        traces.append(trace)

    conn.send(traces)


class IsolatedRunner(Runner):
    """Spawns one subprocess per NPU device for parallel evaluation.

    Each subprocess processes a partition of workloads sequentially.
    Provides full NPU memory isolation between devices.
    """

    def __init__(
        self,
        npu_ids: list[int] | None = None,
        config: BenchmarkConfig | None = None,
    ):
        super().__init__(npu_ids=npu_ids, config=config)

    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solution: Solution,
        case_index: int = 0,
    ) -> Trace:
        """Single-workload evaluation on the first NPU device."""
        if self._work_dir is None:
            self.pre_build(definition, solution, self._config)

        self._append_workload(self._work_dir, workload)

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        proc = ctx.Process(
            target=_isolated_batch_worker,
            args=(
                child_conn, self._config,
                definition, [(workload, case_index, self._npu_ids[0])], solution, str(self._work_dir),
            ),
        )
        proc.start()

        try:
            if parent_conn.poll(self._config.timeout):
                traces = parent_conn.recv()
                trace = traces[0] if traces else Runner._make_error_trace(
                    definition, workload, solution, "RUNTIME_ERROR",
                    "Worker returned no traces",
                )
            else:
                proc.kill()
                trace = Runner._make_error_trace(
                    definition, workload, solution, "TIMEOUT",
                    f"Timeout after {self._config.timeout}s",
                )
        except (EOFError, ConnectionResetError, BrokenPipeError):
            proc.kill()
            trace = Runner._make_error_trace(
                definition, workload, solution, "CRASH",
                "Worker process crashed unexpectedly",
            )

        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
        parent_conn.close()
        child_conn.close()
        return trace

    def run_workloads_batch(
        self,
        definition: Definition,
        workloads: list[Workload],
        solution: Solution,
    ) -> list[Trace]:
        """Evaluate workloads in parallel — one subprocess per NPU device.

        Each subprocess gets a partition of workloads and evaluates
        them sequentially on its assigned device.
        """
        if self._work_dir is None:
            self.pre_build(definition, solution, self._config)

        for wl in workloads:
            self._append_workload(self._work_dir, wl)

        buckets = _partition(workloads, len(self._npu_ids))
        ctx = mp.get_context("spawn")

        workload_to_index = {id(wl): i for i, wl in enumerate(workloads)}

        # Spawn one subprocess per non-empty npu bucket
        processes_and_conns: list[
            tuple[list[tuple[Workload, int, int]], mp.Process, mp.connection.Connection]
        ] = []
        for npu_id, bucket in zip(self._npu_ids, buckets):
            if not bucket:
                continue
            bucket_tasks = [(wl, workload_to_index[id(wl)], npu_id) for wl in bucket]
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_isolated_batch_worker,
                args=(
                    child_conn, self._config,
                    definition, bucket_tasks, solution, str(self._work_dir),
                ),
            )
            proc.start()
            processes_and_conns.append((bucket_tasks, proc, parent_conn))

        # Collect results from each subprocess
        uuid_to_trace: dict[str, Trace] = {}
        for bucket_tasks, proc, conn in processes_and_conns:
            try:
                if conn.poll(self._config.timeout):
                    bucket_traces = conn.recv()
                    for trace in bucket_traces:
                        uuid_to_trace[trace.workload.uuid] = trace
                else:
                    proc.kill()
                    for wl, _, _ in bucket_tasks:
                        if wl.uuid not in uuid_to_trace:
                            uuid_to_trace[wl.uuid] = Runner._make_error_trace(
                                definition, wl, solution, "TIMEOUT",
                                f"Timeout after {self._config.timeout}s",
                            )
            except (EOFError, ConnectionResetError, BrokenPipeError):
                proc.kill()
                for wl, _, _ in bucket_tasks:
                    if wl.uuid not in uuid_to_trace:
                        uuid_to_trace[wl.uuid] = Runner._make_error_trace(
                            definition, wl, solution, "CRASH",
                            "Worker process crashed unexpectedly",
                        )

            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
            conn.close()

        # Reorder to match input workload order
        return [uuid_to_trace[wl.uuid] for wl in workloads]

    def close(self) -> None:
        self._work_dir = None
