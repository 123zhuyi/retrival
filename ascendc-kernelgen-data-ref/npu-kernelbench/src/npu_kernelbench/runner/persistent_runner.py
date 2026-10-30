"""PersistentRunner — long-lived worker processes with per-device queues."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.data.trace import Trace
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.runner.runner import Runner, _partition


@dataclass
class WorkerState:
    """Minimal metadata for a running persistent worker."""
    npu_id: int
    process: mp.Process


def _persistent_worker(
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    npu_id: int,
    config: BenchmarkConfig,
) -> None:
    """Long-lived worker: loop on input_queue, build baseline + evaluate, emit to result_queue."""
    import os

    # Disable torch's torch_npu autoloader so ACL/GE initializes cleanly in this
    # subprocess after ASCEND_RT_VISIBLE_DEVICES has been set.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)
    import torch
    import torch_npu  # noqa: F401
    # Pre-import tbe so that reference code setting torch_npu.npu.conv.allow_hf32
    # can initialise the CANN compiler in this subprocess.
    try:
        import tbe  # noqa: F401
    except Exception:
        pass
    device = "npu:0"

    while True:
        msg = input_queue.get()
        if msg is None:  # SHUTDOWN sentinel
            break

        definition, workload, solution, work_dir, case_index = msg

        try:
            # 1. Load .so and import solution entry point
            solution_fn = Runner._load_solution_fn(work_dir, solution)

            # 2. Point AST scan / kernel signature at staged sources
            config.solution_source_dir = str(Path(work_dir) / "sources")

            # 3. Build baseline inside worker
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

        result_queue.put((npu_id, workload.uuid, trace))


# ── PersistentRunner ────────────────────────────────────────────────────

class PersistentRunner(Runner):
    """Long-lived worker processes with per-device input queues.

    One worker process per NPU device. Each worker loops on its own
    input queue, builds baselines internally, evaluates, and pushes
    trace results to a shared result queue.

    ``run_workloads_batch()`` partitions workloads across devices and
    runs them concurrently. ``run_workload()`` is a single-workload
    adapter using the same queue infrastructure (for single-device /
    debugging use).
    """

    def __init__(
        self,
        npu_ids: list[int] | None = None,
        config: BenchmarkConfig | None = None,
    ):
        super().__init__(npu_ids=npu_ids, config=config)
        self._ctx = mp.get_context("spawn")
        self._queues: list[mp.Queue] = []
        self._result_queue: mp.Queue = self._ctx.Queue()
        self._workers: list[WorkerState] = []
        self._start_workers()

    def _start_workers(self) -> None:
        """Spawn one persistent worker process per NPU device."""
        for npu_id in self._npu_ids:
            input_q: mp.Queue = self._ctx.Queue()
            proc = self._ctx.Process(
                target=_persistent_worker,
                args=(input_q, self._result_queue, npu_id, self._config),
            )
            proc.start()
            self._queues.append(input_q)
            self._workers.append(WorkerState(npu_id=npu_id, process=proc))

    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solution: Solution,
        case_index: int = 0,
    ) -> Trace:
        """Single-workload dispatch. Uses first worker's queue.

        For single-device and debugging. Prefer ``run_workloads_batch``
        for production use.
        """
        if self._work_dir is None:
            self.pre_build(definition, solution, self._config)

        self._append_workload(self._work_dir, workload)
        self._queues[0].put(
            (definition, workload, solution, str(self._work_dir), case_index)
        )

        try:
            _, _, trace = self._result_queue.get(
                timeout=self._config.timeout
            )
            return trace
        except queue.Empty:
            return Runner._make_error_trace(
                definition, workload, solution, "TIMEOUT",
                f"Timeout after {self._config.timeout}s",
            )

    def run_workloads_batch(
        self,
        definition: Definition,
        workloads: list[Workload],
        solution: Solution,
    ) -> list[Trace]:
        """Evaluate workloads in parallel across NPU devices.

        Partitions workloads across workers, pushes all upfront, then
        collects results. Returns traces in the same order as *workloads*.
        """
        if self._work_dir is None:
            self.pre_build(definition, solution, self._config)

        # Append workloads to workload.jsonl (main process, sequential)
        for wl in workloads:
            self._append_workload(self._work_dir, wl)

        # Partition and dispatch (use global workload index for case_index)
        workload_to_index = {id(wl): i for i, wl in enumerate(workloads)}
        buckets = _partition(workloads, len(self._queues))
        for q, bucket in zip(self._queues, buckets):
            for wl in bucket:
                q.put((definition, wl, solution, str(self._work_dir), workload_to_index[id(wl)]))

        # Collect results
        uuid_to_trace: dict[str, Trace] = {}
        for _ in range(len(workloads)):
            try:
                _, uuid, trace = self._result_queue.get(
                    timeout=self._config.timeout
                )
                uuid_to_trace[uuid] = trace
            except queue.Empty:
                # Worker hang or crash — fill remaining with TIMEOUT traces
                for wl in workloads:
                    if wl.uuid not in uuid_to_trace:
                        uuid_to_trace[wl.uuid] = Runner._make_error_trace(
                            definition, wl, solution, "TIMEOUT",
                            f"Timeout after {self._config.timeout}s "
                            f"(or worker crashed)",
                        )
                break

        # Drain any remaining results from the queue after timeout
        # so stale results don't pollute the next batch call.
        if len(uuid_to_trace) < len(workloads):
            while True:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break

        # Reorder to match input workload order
        return [uuid_to_trace[wl.uuid] for wl in workloads]

    def close(self) -> None:
        """Send shutdown sentinel to all workers and join them."""
        for q in self._queues:
            try:
                q.put(None)
            except Exception:
                pass
        for w in self._workers:
            w.process.join(timeout=5)
            if w.process.is_alive():
                w.process.kill()
        # Drain leftover results after workers have shut down
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        self._workers.clear()
        self._queues.clear()
        self._work_dir = None
