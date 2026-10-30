"""Numerical correctness checking for kernel evaluation."""

from __future__ import annotations

import random
import torch
from typing import Optional, Tuple

from npu_kernelbench.data.trace import Correctness
from npu_kernelbench.data.workload import ToleranceSpec
from npu_kernelbench.data.dtypes import is_dtype_integer


# ---------------------------------------------------------------------------
# Ecosystem open-source precision standard thresholds.
#
# Floating-point compute path uses mixed tolerance:
#   |actual - golden| <= atol + rtol * |golden|
#
# A case passes when BOTH:
#   1. matched_ratio = count(matched) / total >= required_matched_ratio (default 0.99)
#   2. max_abs_error <= max_abs_error_limit
#
# The standard dtype-aware defaults below are used unless a workload's
# ToleranceSpec explicitly overrides max_atol / max_rtol / max_error_cap.
# ---------------------------------------------------------------------------

# Default matched-ratio requirement. Workloads may override via
# ToleranceSpec.required_matched_ratio.
REQUIRED_MATCHED_RATIO = 0.9

# (rtol, atol, max_abs_error_limit) per dtype. Values taken from the ecosystem
# open-source precision standard. Unknown dtypes fall back to float32 limits.
DTYPE_PRECISION_LIMITS = {
    "float32": (2.0**-10, 2.0**-16, 1e-2),
    "float": (2.0**-10, 2.0**-16, 1e-2),
    "float16": (2.0**-9, 2.0**-9, 1e-1),
    "half": (2.0**-9, 2.0**-9, 1e-1),
    "bfloat16": (2.0**-6, 2.0**-6, 1.0),
    "hifloat32": (2.0**-9, 2.0**-10, 1e-1),
    "hfloat32": (2.0**-9, 2.0**-10, 1e-1),
    "float8_e4m3": (2.0**-2, 2.0**-4, 1.0),
    "float8_e4m3fn": (2.0**-2, 2.0**-4, 1.0),
    "float8_e5m2": (2.0**-1, 2.0**-3, 1e-1),
    "fp8_e4m3": (2.0**-2, 2.0**-4, 1.0),
    "fp8_e5m2": (2.0**-1, 2.0**-3, 1e-1),
}

_TORCH_DTYPE_LIMITS: Optional[dict] = None


def _get_torch_dtype_limits() -> dict:
    global _TORCH_DTYPE_LIMITS
    if _TORCH_DTYPE_LIMITS is not None:
        return _TORCH_DTYPE_LIMITS
    limits = {
        torch.float32: (2.0**-10, 2.0**-16, 1e-2),
        torch.float16: (2.0**-9, 2.0**-9, 1e-1),
        torch.bfloat16: (2.0**-6, 2.0**-6, 1.0),
    }
    float8_e4m3 = getattr(torch, "float8_e4m3fn", None) or getattr(
        torch, "float8_e4m3", None
    )
    if float8_e4m3 is not None:
        limits[float8_e4m3] = (2.0**-2, 2.0**-4, 1.0)
    float8_e5m2 = getattr(torch, "float8_e5m2fn", None) or getattr(
        torch, "float8_e5m2", None
    )
    if float8_e5m2 is not None:
        limits[float8_e5m2] = (2.0**-1, 2.0**-3, 1e-1)
    _TORCH_DTYPE_LIMITS = limits
    return limits


def _get_precision_limits(data_type) -> tuple[float, float, float]:
    """Return (rtol, atol, max_abs_error_limit) for a dtype.

    Supports torch.dtype and string aliases. Unknown dtypes fall back to the
    float32 limits.
    """
    if isinstance(data_type, str):
        return DTYPE_PRECISION_LIMITS.get(
            data_type.lower(), DTYPE_PRECISION_LIMITS["float32"]
        )
    return _get_torch_dtype_limits().get(
        data_type, DTYPE_PRECISION_LIMITS["float32"]
    )


def _resolve_threshold(
    dtype: torch.dtype,
    tolerance: ToleranceSpec,
) -> tuple[tuple[float, float, float], float]:
    """Resolve thresholds for a dtype.

    Returns (rtol, atol, max_abs_error_limit), required_matched_ratio.

    ``max_atol``/``max_rtol``/``max_error_cap`` on the workload override the
    dtype-aware defaults from the ecosystem precision standard.
    """
    dtype_str = str(dtype).replace("torch.", "")
    default_rtol, default_atol, default_max_abs = _get_precision_limits(dtype_str)
    atol = tolerance.max_atol if "max_atol" in tolerance.model_fields_set else default_atol
    rtol = tolerance.max_rtol if "max_rtol" in tolerance.model_fields_set else default_rtol
    max_abs_error_limit = (
        tolerance.max_error_cap
        if "max_error_cap" in tolerance.model_fields_set and tolerance.max_error_cap is not None
        else default_max_abs
    )
    required_matched_ratio = tolerance.required_matched_ratio
    return (rtol, atol, max_abs_error_limit), required_matched_ratio


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def check_tensor_sanity(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool = False,
) -> Optional[Correctness]:
    """Check for non-finite values and all-zeros output.

    NaN/Inf handling mirrors verify.py: positions must match exactly, and
    any existing Inf values must have matching signs. The workload-level
    ``allow_negative_inf`` extension permits matching -inf pairs.

    Returns a Correctness describing the failure, or None if ok.
    """
    # NaN positions must match exactly
    ref_nan_mask = torch.isnan(ref_tensor)
    sol_nan_mask = torch.isnan(sol_tensor)
    if not torch.equal(ref_nan_mask, sol_nan_mask):
        return Correctness(has_nan=True)

    # Inf positions must match exactly (minus allowed matching -inf pairs)
    ref_inf_mask = torch.isinf(ref_tensor)
    sol_inf_mask = torch.isinf(sol_tensor)
    if allow_negative_inf:
        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))
        ref_inf_mask = ref_inf_mask & ~both_neg_inf
        sol_inf_mask = sol_inf_mask & ~both_neg_inf
    if not torch.equal(ref_inf_mask, sol_inf_mask):
        return Correctness(has_inf=True)

    # If any Inf values remain, their signs must match
    if ref_inf_mask.any():
        if not torch.equal(
            torch.sign(ref_tensor[ref_inf_mask]),
            torch.sign(sol_tensor[sol_inf_mask]),
        ):
            return Correctness(has_inf=True)

    # All-zero output check (only when reference is non-zero)
    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))
    if (
        ref_norm.item() > 0
        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0
    ):
        abs_err = float(ref_norm.item())
        return Correctness(
            max_absolute_error=abs_err,
            max_relative_error=abs_err,
        )

    return None


def _check_bool_or_exact_integral(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[Correctness, bool]:
    """Handle bool and integer types: strict equality required."""
    exact_match = torch.equal(output, reference)
    if exact_match:
        return Correctness(), False
    diff_max = float((output != reference).sum().item())
    return Correctness(max_absolute_error=diff_max, max_relative_error=diff_max), True


def _check_quantized_fp_to_int(
    output: torch.Tensor,  # integer
    reference: torch.Tensor,  # reference (float)
) -> Tuple[Correctness, bool]:
    """Quantized fp->int: allow off-by-1 per Cannbot convention."""
    diff = torch.abs(output.to(torch.int64) - reference.to(torch.int64))
    max_abs = float(diff.max().item())
    exceeds = (diff > 1).any().item()
    return Correctness(max_absolute_error=max_abs), exceeds


def _check_integer_compute(
    output: torch.Tensor,
    reference: torch.Tensor,
    tolerance: ToleranceSpec,
) -> Tuple[Correctness, bool]:
    """Integer compute: strict equality by default.

    int8/uint8 outputs are commonly quantized floating-point results.  When the
    workload sets ``max_atol >= 1`` we allow a per-element off-by-one difference
    (and enforce ``max_error_cap`` when provided), keeping the tolerance knob in
    the workload rather than in the framework.
    """
    diff = torch.abs(output.to(torch.int64) - reference.to(torch.int64))
    max_abs = float(diff.max().item())

    if output.dtype in (torch.int8, torch.uint8):
        atol = tolerance.max_atol
        if max_abs <= atol:
            return Correctness(max_absolute_error=max_abs), False
        if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
            return Correctness(max_absolute_error=max_abs), True
        return Correctness(max_absolute_error=max_abs), True

    if torch.equal(output, reference):
        return Correctness(), False
    return Correctness(max_absolute_error=max_abs, max_relative_error=max_abs), True


def _check_accuracy_mixed_tolerance(
    golden: torch.Tensor,
    actual: torch.Tensor,
    data_type: torch.dtype,
    required_matched_ratio: float,
    atol: float,
    rtol: float,
    max_abs_error_limit: float,
) -> Tuple[bool, dict]:
    """Mixed-tolerance accuracy check aligned with the ecosystem standard.

    Per-element pass condition:
        |actual - golden| <= atol + rtol * |golden|

    Case passes when BOTH:
        1. matched_ratio = count(matched) / total >= required_matched_ratio
        2. max_abs_error <= max_abs_error_limit
    """
    golden_f = golden.float()
    actual_f = actual.float()

    abs_diff = (actual_f - golden_f).abs()
    abs_golden = golden_f.abs()

    matched_mask = abs_diff <= (atol + rtol * abs_golden)

    total_finite = matched_mask.numel()
    matched_count = int(matched_mask.sum().item())
    matched_ratio = matched_count / total_finite if total_finite > 0 else 1.0
    max_abs_diff = float(abs_diff.max().item()) if total_finite > 0 else 0.0
    max_rel_diff = float((abs_diff / (abs_golden + 1e-7)).max().item()) if total_finite > 0 else 0.0

    ratio_ok = matched_ratio >= required_matched_ratio
    max_abs_ok = max_abs_diff <= max_abs_error_limit
    is_pass = ratio_ok and max_abs_ok

    metrics = {
        "matched_ratio": matched_ratio,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "required_matched_ratio": required_matched_ratio,
        "max_abs_error_limit": max_abs_error_limit,
        "atol": atol,
        "rtol": rtol,
        "total_finite": total_finite,
        "matched_count": matched_count,
        "checks": {
            "required_matched_ratio": ratio_ok,
            "max_abs_error_limit": max_abs_ok,
        },
    }
    return is_pass, metrics


def _infer_input_type(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> str:
    """Infer input_type for decision matrix.

    npu-kernelbench only has the output tensors at comparison time, so we
    approximate verify.py's input_type classification from the reference dtype:
    - float (any floating point / complex) -> 'float'
    - integer / bool -> 'int'
    This matches verify.py's two-class split used for output-int routing.
    """
    if reference.dtype.is_floating_point or reference.dtype.is_complex:
        return "float"
    return "int"


def compute_error_stats(
    output: torch.Tensor,
    reference: torch.Tensor,
    tolerance: ToleranceSpec,
) -> Tuple[Correctness, bool]:
    """Compute numerical error between output and reference.

    Aligns with the ecosystem open-source precision standard:
      - shape/dtype/NaN/Inf checks are performed upstream by the evaluator
      - bool output: torch.equal strict equality
      - int output + int-like input: strict equality
      - int output + float input: |diff| <= 1
      - float output: mixed tolerance with
          |actual - golden| <= atol + rtol * |golden|
        case passes when BOTH:
          1. matched_ratio >= required_matched_ratio (default 0.99)
          2. max_abs_error <= max_abs_error_limit

    The workload's ToleranceSpec can override the dtype-aware defaults for
    ``max_atol``, ``max_rtol``, ``max_error_cap`` and ``required_matched_ratio``.

    Returns (correctness, exceeds_tolerance).
    """
    output_dtype = output.dtype
    ref_dtype = reference.dtype

    # Path 1: bool output -> strict equality
    if output_dtype == torch.bool or ref_dtype == torch.bool:
        return _check_bool_or_exact_integral(output, reference)

    # Path 2: integer output -> route by inferred input_type
    if is_dtype_integer(output_dtype) and is_dtype_integer(ref_dtype):
        input_type = _infer_input_type(output, reference)
        if input_type == "float":
            return _check_quantized_fp_to_int(output, reference)
        return _check_integer_compute(output, reference, tolerance)

    # Path 3: mixed int/float (quantized) -> |diff| <= 1
    if is_dtype_integer(output_dtype) != is_dtype_integer(ref_dtype):
        return _check_quantized_fp_to_int(output, reference)

    # Path 4: floating-point output -> mixed tolerance two-condition check
    (rtol, atol, max_abs_error_limit), required_matched_ratio = _resolve_threshold(
        ref_dtype, tolerance
    )

    allow_neg_inf = tolerance.allow_negative_inf
    infs_nans = check_tensor_sanity(output, reference, allow_negative_inf=allow_neg_inf)
    if infs_nans is not None:
        return infs_nans, True

    finite_mask = torch.isfinite(output) & torch.isfinite(reference)
    golden = reference[finite_mask]
    actual = output[finite_mask]

    if golden.numel() == 0:
        return Correctness(), False

    if actual.dtype != golden.dtype:
        actual = actual.to(golden.dtype)

    is_pass, metrics = _check_accuracy_mixed_tolerance(
        golden,
        actual,
        golden.dtype,
        required_matched_ratio,
        atol,
        rtol,
        max_abs_error_limit,
    )
    exceeds_tol = not is_pass

    correctness = Correctness(
        max_absolute_error=metrics["max_abs_diff"],
        max_relative_error=metrics["max_rel_diff"],
        extra=metrics,
    )
    return correctness, exceeds_tol
