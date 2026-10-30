"""Evaluator ABC — base class for kernel-type-specific evaluation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Complex, Integral, Real
from typing import Any, Callable, Optional
import torch
from npu_kernelbench.evaluator.anti_hack import (
    run_ast_scan,
    extract_kernel_signatures,
    detect_kernel_signature_mismatch,
    detect_output_caching,
    detect_monkey_patch,
    detect_clock_anomaly,
    _format_evidence,
)
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.data.trace import (
    Correctness, Performance, Evaluation, EvaluationStatus,
    Environment, Trace,
)
from npu_kernelbench.cli.config import BenchmarkConfig


# ---------- output shape/dtype helpers (multi-tensor friendly) ----------

def _is_list_output(v: Any) -> bool:
    """True when an output value is a list/tuple collection."""
    return isinstance(v, (list, tuple))


def _as_list(v: Any) -> list:
    """Wrap a single output into a list; pass lists/tuples through."""
    return list(v) if _is_list_output(v) else [v]


def _scalar_kind(v: Any) -> type:
    """Return a stable dtype-like category for Python scalar outputs."""
    if isinstance(v, bool):
        return bool
    if isinstance(v, Integral):
        return int
    if isinstance(v, Real):
        return float
    if isinstance(v, Complex):
        return complex
    return type(v)


def _shapes_match(sol: Any, ref: Any) -> bool:
    """Compare tensor shapes and recursively accept scalar leaves."""
    if _is_list_output(sol) or _is_list_output(ref):
        sl, rl = _as_list(sol), _as_list(ref)
        if len(sl) != len(rl):
            return False
        return all(_shapes_match(s, r) for s, r in zip(sl, rl))
    if isinstance(sol, torch.Tensor) or isinstance(ref, torch.Tensor):
        return (
            isinstance(sol, torch.Tensor)
            and isinstance(ref, torch.Tensor)
            and sol.shape == ref.shape
        )
    return True


def _dtypes_match(sol: Any, ref: Any) -> bool:
    """Compare tensor dtypes and scalar type categories recursively."""
    if _is_list_output(sol) or _is_list_output(ref):
        sl, rl = _as_list(sol), _as_list(ref)
        if len(sl) != len(rl):
            return False
        return all(_dtypes_match(s, r) for s, r in zip(sl, rl))
    if isinstance(sol, torch.Tensor) or isinstance(ref, torch.Tensor):
        return (
            isinstance(sol, torch.Tensor)
            and isinstance(ref, torch.Tensor)
            and sol.dtype == ref.dtype
        )
    return _scalar_kind(sol) is _scalar_kind(ref)


def _shape_repr(v: Any) -> Any:
    """Return human-readable shapes for tensors, collections, and scalars."""
    if _is_list_output(v):
        return [_shape_repr(item) for item in v]
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return "scalar"


def _dtype_repr(v: Any) -> Any:
    """Return human-readable dtypes for tensors, collections, and scalars."""
    if _is_list_output(v):
        return [_dtype_repr(item) for item in v]
    if isinstance(v, torch.Tensor):
        return v.dtype
    return _scalar_kind(v).__name__


# ---------- baseline handle ----------

@dataclass
class BaselineHandle:
    """Pre-computed reference data shared across evaluations.

    Built once per (definition, workload) pair and reused across all
    solutions evaluating against the same workload.
    """
    definition_name: str
    device: str
    workload_uuid: str
    inputs: list = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    mean_latency_ms: Optional[float] = None


@dataclass
class PerfDiagnostics:
    """Additional performance data needed by anti-hack checks."""
    durations: list[float] = field(default_factory=list)
    trace_dir: str = ""


# ---------- abstract evaluator ----------

class Evaluator(ABC):
    """Abstract evaluator for a specific kernel type.

    Subclasses implement kernel-type-specific correctness checking.
    The template method evaluate() orchestrates the full pipeline.
    """

    @abstractmethod
    def can_evaluate(self, definition: Definition) -> bool:
        """Return True if this evaluator handles the given definition."""
        ...

    @abstractmethod
    def build_baseline(
        self,
        definition: Definition,
        workload: Workload,
        device: str,
        config: BenchmarkConfig,
    ) -> BaselineHandle:
        """Generate inputs, run reference implementation, capture outputs.

        The baseline is cached by the caller (Runner) and reused across
        multiple solution evaluations against the same workload.
        """
        ...

    @abstractmethod
    def check_correctness(
        self,
        definition: Definition,
        workload: Workload,
        baseline: BaselineHandle,
        solution_outputs: dict[str, Any],
    ) -> tuple[Correctness, bool]:
        """Check solution outputs against baseline.

        Returns (correctness_metrics, is_pass).
        is_pass=True means outputs are correct within tolerance.
        """
        ...

    def eval_performance(
        self,
        solution_fn: Callable,
        inputs: list,
        definition: Definition,
        config: BenchmarkConfig,
    ) -> tuple[Performance, PerfDiagnostics]:
        """Measure solution execution time.

        Default: use timing.profile_kernel(). Returns both the public
        ``Performance`` metrics and internal ``PerfDiagnostics`` needed
        by anti-hack checks.
        """
        from npu_kernelbench.evaluator.performance import profile_kernel

        input_args = tuple(inputs)
        timing = profile_kernel(
            solution_fn,
            input_args,
            npu_id=config.npu_id,
            warmup_runs=config.warmup_runs,
            iterations=config.iterations,
            profile_dir=config.profile_dir or f"{config.work_dir}/profile",
        )
        perf = Performance(latency_ms=timing.mean_ms)
        diagnostics = PerfDiagnostics(
            durations=timing.durations,
            trace_dir=timing.trace_dir,
        )
        return perf, diagnostics

    def evaluate(
        self,
        definition: Definition,
        workload: Workload,
        solution_fn: Callable,
        baseline: BaselineHandle,
        config: BenchmarkConfig,
        solution: Optional[Solution] = None,
    ) -> Trace:
        """Template method: baseline -> correctness -> performance -> anti-hack -> Trace.

        Orchestrates the full evaluation pipeline. Subclasses typically
        only override can_evaluate(), build_baseline(), check_correctness(),
        and optionally eval_performance().
        """
        from datetime import datetime, timezone
        from pathlib import Path

        hw_name = _get_hardware_name()

        # ---------- L0: Anti-hack — static AST scan (if enabled) ----------
        hacks: list[str] = []
        if config.mode != "correctness_only":
            if "ast_scan" in config.enable_anti_hack and config.solution_source_dir:
                source_dir = Path(config.solution_source_dir)
                if source_dir.exists():
                    py_files = list(source_dir.glob("*.py"))
                    if py_files:
                        hacks.extend(run_ast_scan(py_files))

        def _join_hacks(msg: str) -> str:
            if hacks:
                return "\n".join(hacks) + "\n" + msg
            return msg

        # ---------- 1. Run solution ----------
        try:
            input_args = tuple(baseline.inputs)
            sol_raw = solution_fn(*input_args)
            if not isinstance(sol_raw, (tuple, list)):
                sol_raw = [sol_raw]
            sol_outputs_list = list(sol_raw)
        except Exception as e:
            return Trace(
                definition=definition.name,
                workload=workload,
                solution=getattr(solution_fn, "__name__", "unknown"),
                evaluation=Evaluation(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    environment=Environment(hardware=hw_name),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    log=_join_hacks(str(e)),
                ),
            )

        # Normalize to dict. A single named output may map to a list/tuple of
        # tensors (e.g. Split → one name "chunks" but N chunk tensors). When the
        # name count and tensor count disagree and there is exactly one name,
        # keep the whole list under that name so downstream list-aware checks
        # compare element-wise. Single-tensor outputs are unaffected.
        out_names = list(definition.outputs.keys())
        if len(out_names) == len(sol_outputs_list):
            sol_outputs = dict(zip(out_names, sol_outputs_list))
        elif len(out_names) == 1:
            sol_outputs = {out_names[0]: sol_outputs_list}
        else:
            return Trace(
                definition=definition.name,
                workload=workload,
                solution=getattr(solution_fn, "__name__", "unknown"),
                evaluation=Evaluation(
                    status=EvaluationStatus.INCORRECT_SHAPE,
                    environment=Environment(hardware=hw_name),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    log=_join_hacks(
                        f"Expected {len(out_names)} outputs "
                        f"{out_names}, got {len(sol_outputs_list)}"
                    ),
                ),
            )

        # ---------- 2. Shape check ----------
        ref_outputs = baseline.outputs
        for name in definition.outputs:
            if not _shapes_match(sol_outputs[name], ref_outputs[name]):
                return Trace(
                    definition=definition.name,
                    workload=workload,
                    solution=getattr(solution_fn, "__name__", "unknown"),
                    evaluation=Evaluation(
                        status=EvaluationStatus.INCORRECT_SHAPE,
                        environment=Environment(hardware=hw_name),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        log=_join_hacks(
                            f"Output '{name}': sol shape {_shape_repr(sol_outputs[name])} "
                            f"!= ref shape {_shape_repr(ref_outputs[name])}"
                        ),
                    ),
                )

        # ---------- 3. Dtype check ----------
        for name in definition.outputs:
            if not _dtypes_match(sol_outputs[name], ref_outputs[name]):
                return Trace(
                    definition=definition.name,
                    workload=workload,
                    solution=getattr(solution_fn, "__name__", "unknown"),
                    evaluation=Evaluation(
                        status=EvaluationStatus.INCORRECT_DTYPE,
                        environment=Environment(hardware=hw_name),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        log=_join_hacks(
                            f"Output '{name}': sol dtype "
                            f"{_dtype_repr(sol_outputs[name])} != ref dtype "
                            f"{_dtype_repr(ref_outputs[name])}"
                        ),
                    ),
                )

        # ---------- 4. Correctness check (subclass) ----------
        correctness, is_pass = self.check_correctness(
            definition, workload, baseline, sol_outputs
        )

        if not is_pass:
            return Trace(
                definition=definition.name,
                workload=workload,
                solution=getattr(solution_fn, "__name__", "unknown"),
                evaluation=Evaluation(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    environment=Environment(hardware=hw_name),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    correctness=correctness,
                    log=_join_hacks(""),
                ),
            )

        # ---------- 5. Performance (skip in correctness_only mode) ----------
        perf = Performance(latency_ms=0.0)
        diagnostics = PerfDiagnostics(durations=[], trace_dir="")
        if config.mode != "correctness_only":
            perf, diagnostics = self.eval_performance(
                solution_fn, baseline.inputs, definition, config
            )
            if config.perf_reference_source == "baseline" and workload.perf_baseline:
                ref_latency = workload.perf_baseline.latency_ms
            else:
                ref_latency = baseline.mean_latency_ms
                if ref_latency is None and workload.perf_baseline:
                    ref_latency = workload.perf_baseline.latency_ms
            if ref_latency is not None:
                perf.reference_latency_ms = ref_latency
                perf.speedup_factor = ref_latency / max(perf.latency_ms, 1e-9)

        # ---------- 6. Anti-hack L1/L2 checks (skip in correctness_only mode) ----------
        if config.mode != "correctness_only":
            # 6a. L1: kernel signature mismatch
            if "kernel_signature" in config.enable_anti_hack and config.solution_source_dir:
                source_dir = Path(config.solution_source_dir)
                profile_dir = Path(config.profile_dir or f"{config.work_dir}/profile")
                try:
                    hack_logs = detect_kernel_signature_mismatch(
                        source_dir, profile_dir
                    )
                    hacks.extend(hack_logs)
                except Exception as e:
                    import logging
                    logging.warning("Kernel signature check failed (non-fatal): %s", e, exc_info=True)

            # 6b. L2: output caching
            if "output_caching" in config.enable_anti_hack:
                try:
                    from npu_kernelbench.evaluator.generator import gen_inputs

                    def _input_generator(seed: int) -> tuple:
                        return tuple(
                            gen_inputs(
                                definition, workload, baseline.device, seed, case_index=config.case_index
                            )
                        )

                    result = detect_output_caching(solution_fn, _input_generator)
                    if result.detected:
                        hacks.append(
                            f"Reward hack ({result.hack_type}): {_format_evidence(result.evidence)}"
                        )
                except Exception as e:
                    import logging
                    logging.warning("Output caching check failed (non-fatal): %s", e, exc_info=True)

            # 6c. L2: monkey patch detection
            if "monkey_patch" in config.enable_anti_hack and definition.reference_hash:
                result = detect_monkey_patch(definition.reference, definition.reference_hash)
                if result.detected:
                    hacks.append(
                        f"Reward hack ({result.hack_type}): {_format_evidence(result.evidence)}"
                    )

            # 6d. L2: clock anomaly detection
            if "clock_anomaly" in config.enable_anti_hack and diagnostics.durations:
                result = detect_clock_anomaly(diagnostics.durations)
                if result.detected:
                    hacks.append(
                        f"Reward hack ({result.hack_type}): {_format_evidence(result.evidence)}"
                    )

            # If any hack was detected, return REWARD_HACK
            if hacks:
                return Trace(
                    definition=definition.name,
                    workload=workload,
                    solution=getattr(solution_fn, "__name__", "unknown"),
                    evaluation=Evaluation(
                        status=EvaluationStatus.REWARD_HACK,
                        environment=Environment(hardware=hw_name),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        log="\n".join(hacks),
                    ),
                )

        # ---------- 7. All checks passed ----------
        return Trace(
            definition=definition.name,
            workload=workload,
            solution=getattr(solution_fn, "__name__", "unknown"),
            evaluation=Evaluation(
                status=EvaluationStatus.PASSED,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                correctness=correctness,
                performance=perf,
            ),
        )


def _get_hardware_name() -> str:
    """Detect NPU hardware name at runtime."""
    import os
    try:
        if torch.npu.is_available():
            name = torch.npu.get_device_name()
            if name:
                return name
    except Exception:
        pass
    return os.environ.get("NPU_HARDWARE_NAME", "Ascend_Unknown")