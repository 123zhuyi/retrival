"""DefaultEvaluator — general-purpose numerical correctness checker."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import CustomInput, SafetensorsInput, Workload
from npu_kernelbench.data.trace import Correctness
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.evaluator.correctness import (
    set_seed,
    check_tensor_sanity,
    compute_error_stats,
)
from npu_kernelbench.evaluator.evaluator import (
    BaselineHandle,
    Evaluator,
    _as_list,
    _is_list_output,
)
from npu_kernelbench.evaluator.generator import gen_inputs, load_safetensors


class DefaultEvaluator(Evaluator):
    """Handles general-purpose kernels: GEMM, RMSNorm, RoPE, etc.

    Uses dtype-aware MERE/MARE checks aligned with verify.py.
    Always returns True from can_evaluate() — this is the catch-all evaluator.
    """

    def can_evaluate(self, definition: Definition) -> bool:
        return True

    def build_baseline(
        self,
        definition: Definition,
        workload: Workload,
        device: str,
        config: BenchmarkConfig,
    ) -> BaselineHandle:
        """Generate inputs, run reference, return baseline."""
        set_seed(config.seed)

        device_obj = torch.device(device)

        # Determine if safetensors or custom inputs are needed
        has_safetensors = any(
            isinstance(s, SafetensorsInput) for s in workload.inputs.values()
        )
        has_custom = any(
            isinstance(s, CustomInput) for s in workload.inputs.values()
        )

        # Load safetensors
        safe_tensors = (
            load_safetensors(definition, workload) if has_safetensors else {}
        )

        # Prepare custom inputs function
        custom_fn = None
        if has_custom:
            custom_fn = _load_custom_inputs_fn(definition, device_obj)

        # Generate inputs as ordered list
        inputs = gen_inputs(
            definition, workload, device, config.seed,
            safe_tensors=safe_tensors, custom_inputs_fn=custom_fn,
            case_index=config.case_index,
        )

        # Run reference — stores outputs keyed by definition output names
        ref_outputs = _run_reference(
            definition.reference, inputs, list(definition.outputs.keys()), device=device_obj
        )

        # Optionally time reference (skip in correctness_only mode)
        ref_latency = None
        if config.mode != "correctness_only" and config.benchmark_reference:
            from npu_kernelbench.evaluator.performance import profile_kernel

            def _ref_fn(*args):
                return _run_reference(definition.reference, list(args), device=device_obj)

            timing = profile_kernel(
                _ref_fn,
                tuple(inputs),
                npu_id=config.npu_id,
                warmup_runs=config.warmup_runs,
                iterations=config.iterations,
                profile_dir=config.profile_dir or f"{config.work_dir}/profile",
            )
            ref_latency = timing.mean_ms

        return BaselineHandle(
            definition_name=definition.name,
            device=device,
            workload_uuid=workload.uuid,
            inputs=inputs,
            outputs=ref_outputs,
            mean_latency_ms=ref_latency,
        )

    def check_correctness(
        self,
        definition: Definition,
        workload: Workload,
        baseline: BaselineHandle,
        solution_outputs: dict[str, Any],
    ) -> tuple[Correctness, bool]:
        """Per-element numerical error check using tolerance from workload.

        Iterates over output tensors by definition output order. Compares
        each solution output against the corresponding baseline output.
        Handles both single tensors and single-named list/tuple outputs
        (e.g. Split "chunks", Nonzero as_tuple=True) element-wise.
        """
        tol = workload.tolerance
        max_abs = 0.0
        max_rel = 0.0

        for i, name in enumerate(definition.outputs.keys()):
            sol = solution_outputs[name]
            ref = list(baseline.outputs.values())[i]

            try:
                output_pairs = list(_iter_output_pairs(sol, ref))
            except ValueError:
                correctness = Correctness(
                    max_absolute_error=float("inf"),
                    max_relative_error=float("inf"),
                )
                return _with_output_context(correctness, name), False

            for s, r in output_pairs:
                if isinstance(s, torch.Tensor) and isinstance(r, torch.Tensor):
                    sanity = check_tensor_sanity(
                        s, r, allow_negative_inf=tol.allow_negative_inf
                    )
                    if sanity is not None:
                        return _with_output_context(sanity, name), False
                    correctness, exceeds = compute_error_stats(s, r, tol)
                elif isinstance(s, torch.Tensor) or isinstance(r, torch.Tensor):
                    correctness = Correctness(
                        max_absolute_error=float("inf"),
                        max_relative_error=float("inf"),
                    )
                    return _with_output_context(correctness, name), False
                else:
                    correctness, exceeds = _compare_scalar_outputs(s, r, tol)

                max_abs = max(max_abs, correctness.max_absolute_error)
                max_rel = max(max_rel, correctness.max_relative_error)
                if exceeds:
                    return _with_output_context(correctness, name), False

        return Correctness(
            max_absolute_error=max_abs, max_relative_error=max_rel
        ), True


def _with_output_context(correctness: Correctness, name: str) -> Correctness:
    """Attach the first failing definition output without dropping error stats."""
    extra = dict(correctness.extra or {})
    extra["output_name"] = name
    return correctness.model_copy(update={"extra": extra})


def _iter_output_pairs(sol: Any, ref: Any):
    """Yield corresponding tensor/scalar leaves from nested output collections."""
    if _is_list_output(sol) or _is_list_output(ref):
        sol_items = _as_list(sol)
        ref_items = _as_list(ref)
        if len(sol_items) != len(ref_items):
            raise ValueError("Output collection lengths do not match")
        for sol_item, ref_item in zip(sol_items, ref_items):
            yield from _iter_output_pairs(sol_item, ref_item)
        return
    yield sol, ref


def _compare_scalar_outputs(sol: Any, ref: Any, tolerance) -> tuple[Correctness, bool]:
    """Compare scalar metadata exactly, with tolerance for real-valued floats."""
    if isinstance(sol, bool) or isinstance(ref, bool):
        is_match = sol is ref
    elif isinstance(sol, Integral) and isinstance(ref, Integral):
        is_match = sol == ref
    elif isinstance(sol, Real) and isinstance(ref, Real):
        sol_value = float(sol)
        ref_value = float(ref)
        if math.isnan(sol_value) or math.isnan(ref_value):
            is_match = math.isnan(sol_value) and math.isnan(ref_value)
        elif math.isinf(sol_value) or math.isinf(ref_value):
            is_match = sol_value == ref_value
        else:
            is_match = math.isclose(
                sol_value,
                ref_value,
                rel_tol=tolerance.max_rtol,
                abs_tol=tolerance.max_atol,
            )
    else:
        is_match = sol == ref

    if is_match:
        return Correctness(), False

    try:
        abs_error = abs(float(sol) - float(ref))
        rel_error = abs_error / max(abs(float(ref)), 1e-7)
    except (TypeError, ValueError):
        abs_error = float("inf")
        rel_error = float("inf")
    if math.isnan(abs_error) or math.isnan(rel_error):
        abs_error = float("inf")
        rel_error = float("inf")
    return Correctness(
        max_absolute_error=abs_error,
        max_relative_error=rel_error,
    ), True


def _run_reference(
    reference_code: str,
    inputs: list,
    output_names: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute reference implementation and return output values.

    When output_names is provided, returns a dict keyed by output name.
    Otherwise falls back to positional index strings ('0', '1', ...).

    If ``device`` is provided, returned tensors are moved to that device so
    downstream comparisons against NPU solution outputs do not fail on
    device mismatches.  Reference code ported from CPU-oriented frameworks
    often omits the ``device`` argument when creating intermediate tensors.
    """
    local_ns: dict = {}
    exec(reference_code, local_ns)
    run_fn = local_ns["run"]
    result = run_fn(*inputs)
    if not isinstance(result, (tuple, list)):
        result = [result]

    def _to_device(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=False)
        if isinstance(v, (list, tuple)):
            return type(v)(_to_device(t) for t in v)
        return v

    if device is not None:
        result = [_to_device(t) for t in result]

    if output_names and len(output_names) == len(result):
        return dict(zip(output_names, result))
    if output_names and len(output_names) == 1:
        # Single named output that is itself a list/tuple of tensors
        # (e.g. Split "chunks", Nonzero as_tuple=True): keep the whole
        # collection under the one name for list-aware downstream checks.
        return {output_names[0]: list(result)}
    return {str(i): t for i, t in enumerate(result)}


def _load_custom_inputs_fn(
    definition: Definition,
    device: torch.device,
):
    """Extract custom_inputs_entrypoint from definition.reference and return a callable.

    Returns a function with signature
    (resolved_axes: dict, device: torch.device) -> dict[str, Tensor].
    """
    entrypoint = definition.custom_inputs_entrypoint
    if not entrypoint:
        raise RuntimeError(
            f"Definition '{definition.name}' has CustomInput but "
            f"no custom_inputs_entrypoint"
        )

    local_ns: dict = {}
    exec(definition.reference, local_ns)

    if entrypoint not in local_ns:
        raise RuntimeError(
            f"custom_inputs_entrypoint '{entrypoint}' not found in "
            f"reference code for definition '{definition.name}'"
        )

    custom_fn = local_ns[entrypoint]

    def wrapper(resolved_axes: dict, dev: torch.device) -> dict:
        return custom_fn(resolved_axes, dev)

    return wrapper
