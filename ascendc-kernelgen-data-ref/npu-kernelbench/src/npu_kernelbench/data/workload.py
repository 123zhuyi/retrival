"""Workload specification — defines input tensors and tolerances for a single benchmark run."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import Field, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class DistributionType(str, Enum):
    """Distribution strategy for float random tensors."""

    MIXED = "mixed"
    """50% uniform(-5,5) + 50% normal with random mu~U(-5,5) and sigma~U(0.1,2).
    The uniform branch defaults to low=-5, high=5 (overridable via low/high);
    the normal branch samples mu/sigma per case (overridable via
    mu_low/mu_high/sigma_low/sigma_high)."""

    RANDN = "randn"
    """Standard normal N(0,1)."""

    UNIFORM = "uniform"
    """Uniform(low, high), default uniform(-5,5); low/high overridable via dist_kwargs."""

    NORMAL = "normal"
    """Normal N(mu, sigma) with random mu~U(-5,5) and sigma~U(0.1,2);
    ranges overridable via mu_low/mu_high/sigma_low/sigma_high."""

    ZEROS = "zeros"
    """All zeros tensor (for m/v in AdamW, bias in GroupNormSwish, etc)."""

    ONES = "ones"
    """All ones tensor (for weight in GroupNormSwish, etc)."""

    UNIFORM_SIGNED = "uniform_signed"
    """Uniform with random sign flip: uniform_(low,high) * rand_sign,
    default low=0.3, high=3.0 (value range [-3,-0.3] U [0.3,3], no near-zero
    values). Used for AdamW grad."""


class RandomInput(BaseModelWithDocstrings):
    """Random input generation descriptor."""

    type: Literal["random"] = "random"
    dtype: Optional[str] = None
    """Override the default dtype from Definition. When None, use Definition's dtype."""
    shape: Optional[list[int]] = None
    """Concrete shape for this tensor. When provided, bypasses symbolic axis resolution."""
    range: Optional[list[int]] = None
    """Range [low, high) for integer random generation (high exclusive)."""
    distribution: Optional[DistributionType] = None
    """Float tensor distribution strategy. None defaults to MIXED (50/50 uniform+normal)."""
    mu_low: Optional[float] = None
    """Lower bound for mu (mean) in normal distribution. Default varies by context."""
    mu_high: Optional[float] = None
    """Upper bound for mu (mean) in normal distribution. Default varies by context."""
    sigma_low: Optional[float] = None
    """Lower bound for sigma (std) in normal distribution. Default varies by context."""
    sigma_high: Optional[float] = None
    """Upper bound for sigma (std) in normal distribution. Default varies by context."""
    low: Optional[float] = None
    high: Optional[float] = None
    fill: Optional[str] = None
    """Fill value: 'inf', '-inf', 'nan', 'zero'. When set, generate torch.full instead of random."""

class ScalarInput(BaseModelWithDocstrings):
    """Scalar literal input specification."""

    type: Literal["scalar"] = "scalar"
    value: Union[int, float, bool, str, list, None]


class CustomInput(BaseModelWithDocstrings):
    """Custom input generated from reference code entrypoint."""

    type: Literal["custom"] = "custom"


class SafetensorsInput(BaseModelWithDocstrings):
    """Input loaded from safetensors file."""

    type: Literal["safetensors"] = "safetensors"
    path: NonEmptyString
    tensor_key: NonEmptyString


class TensorListInput(BaseModelWithDocstrings):
    """Variable-length list of tensors sharing the same dtype."""

    type: Literal["tensor_list"] = "tensor_list"
    dtype: str
    """Data type shared by all tensors in the list."""
    shapes: list[list[int]]
    """Shape of each tensor in the list."""


InputSpec = Union[RandomInput, SafetensorsInput, ScalarInput, CustomInput, TensorListInput]
"""Union type for all input specification types."""


class ToleranceSpec(BaseModelWithDocstrings):
    """Tolerance specification for correctness checking.

    For floating-point tensors the evaluator uses dtype-aware MERE/MARE
    thresholds aligned with verify.py.  The ``required_matched_ratio`` field
    provides an additional per-element pass-ratio guard: a workload may set a
    value other than the default, and if unset the Cannbot default of 0.9 is
    used.  ``max_atol``/``max_rtol`` are retained for backward compatibility.
    """

    max_atol: float = Field(default=1e-2)
    """Deprecated for floating-point checks. Kept for backward compatibility."""
    max_rtol: float = Field(default=1e-2)
    """Deprecated for floating-point checks. Kept for backward compatibility."""
    required_matched_ratio: float = Field(default=0.9)
    """Minimum ratio of elements that must pass the per-element bound."""
    max_error_cap: Optional[float] = Field(default=None)
    """Hard ceiling on maximum absolute error. Still enforced."""
    allow_negative_inf: bool = Field(default=False)
    """When True, matching -inf values are treated as correct."""

class PerfBaseline(BaseModelWithDocstrings):
    """性能基线数据。"""

    hardware: NonEmptyString
    """芯片/硬件型号。如 'ascend910b', 'ascend910_93'。"""

    latency_ms: float = Field(ge=0.0)
    """端到端延迟（毫秒）。"""

    notes: Optional[str] = None
    """测量条件备注。"""

class Workload(BaseModelWithDocstrings):
    """Concrete workload configuration for benchmarking."""

    uuid: NonEmptyString
    """Unique identifier for this workload configuration."""
    axes: Optional[dict[str, NonNegativeInt]] = None
    """Dictionary mapping axis names to their concrete integer values. None for NPU-style workloads."""
    inputs: dict[str, InputSpec]
    """Dictionary mapping input names to their data specifications."""
    tolerance: ToleranceSpec = Field(default=ToleranceSpec())
    """Tolerance specification for this workload."""
    perf_baseline: Optional[PerfBaseline] = None
    """性能基线。
    非 None 时参与评分：
      speedup = perf_baseline.latency_ms / solution_latency_ms
    为 None 表示仅做正确性验证，不参与性能对比。
    """

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Workload":
        custom_inputs = [
            name
            for name, inp in self.inputs.items()
            if isinstance(inp, CustomInput)
        ]
        non_custom_inputs = [
            name
            for name, inp in self.inputs.items()
            if not isinstance(inp, CustomInput)
        ]
        if len(custom_inputs) > 0 and len(non_custom_inputs) > 0:
            raise ValueError(
                f"Cannot mix custom and non-custom inputs. "
                f"Custom: {custom_inputs}. Non-custom: {non_custom_inputs}"
            )
        for name, inp in self.inputs.items():
            if not isinstance(inp, RandomInput) and getattr(inp, "distribution", None) is not None:
                raise ValueError(
                    f"Input '{name}' has 'distribution' but is not a RandomInput"
                )
        return self

    def get_scalar_inputs(self) -> dict[str, Union[int, float, bool, str]]:
        return {
            name: inp.value
            for name, inp in self.inputs.items()
            if isinstance(inp, ScalarInput)
        }