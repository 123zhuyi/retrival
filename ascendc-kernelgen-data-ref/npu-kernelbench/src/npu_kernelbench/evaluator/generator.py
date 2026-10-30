"""Input tensor generation for benchmark workloads.

Generates concrete torch.Tensor inputs from Definition + Workload specs,
respecting InputSpec types: RandomInput, ScalarInput, SafetensorsInput, CustomInput.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

import torch
from npu_kernelbench.data.dtypes import dtype_str_to_torch_dtype
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import (
    Workload,
)


def _rand_tensor(
    shape: list[int],
    dtype: torch.dtype,
    device: torch.device,
    range_spec: Optional[list[int]] = None,
    distribution: Optional[str] = None,
    **dist_kwargs: Any,
) -> torch.Tensor:
    """Generate a random tensor with dtype-aware distribution.

    The default distribution for floating-point types is ``mixed``
    (50% uniform[-5,5] + 50% normal with mu∼U[-5,5], sigma∼U[0.1,2]).
    Override per-tensor by setting ``distribution`` in the workload JSON.

    Args:
        shape: Tensor dimensions.
        dtype: Target torch dtype.
        device: Target device.
        range_spec: Optional [low, high] for integer types (inclusive).
        distribution: ``"uniform"``, ``"randn"``, ``"mixed"``,
            ``"normal"``, ``"zeros"``, ``"ones"``, ``"uniform_signed"``,
            or ``None`` (defaults to ``"mixed"``).
            * "uniform" — uniform in [low, high] (``dist_kwargs``) or [-5,5].
            * "randn"   — standard normal ``N(0,1)`` (floats only).
            * "normal"  —  ``N(mu, sigma)``  with random mu∈[-5,5],
               sigma∈[0.1,2] (floats only).
            * "mixed"   — 50 % uniform[-5,5], 50 % normal with random
               mu∈[-5,5], sigma∈[0.1,2] (floats only).
            * "zeros"   — all zeros tensor.
            * "ones"    — all ones tensor.
            * "uniform_signed" — uniform(low,high) * random sign (±1).
            * None      — same as ``"mixed"``.
        **dist_kwargs: Extra keyword args for distribution tuning, e.g.
            ``low``, ``high``, ``mu_low``, ``mu_high``, ``sigma_low``,
            ``sigma_high``.

    Returns:
        Random tensor of requested shape and dtype.

    Raises:
        ValueError: If dtype is not supported for random generation.
    """
    fill_value = dist_kwargs.get("fill")
    if fill_value is not None:
        fill_map = {
            "inf": float("inf"),
            "-inf": float("-inf"),
            "nan": float("nan"),
            "zero": 0.0,
        }
        if fill_value not in fill_map:
            raise ValueError(
                f"Unsupported fill value '{fill_value}'. "
                f"Supported: {list(fill_map.keys())}"
            )
        return torch.full(shape, fill_map[fill_value], dtype=dtype, device=device)

    is_float = dtype in (torch.float32, torch.float16, torch.bfloat16)
    is_lp_float = dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float4_e2m1fn_x2,
    )

    if is_float:
        # Default: mixed (majority of migrated operators use this)
        if distribution is None:
            distribution = "mixed"

        if distribution == "uniform":
            low = float(dist_kwargs.get("low", -5.0))
            high = float(dist_kwargs.get("high", 5.0))
            return torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
        if distribution == "zeros":
            return torch.zeros(shape, dtype=dtype, device=device)
        if distribution == "ones":
            return torch.ones(shape, dtype=dtype, device=device)
        if distribution == "uniform_signed":
            low = float(dist_kwargs.get("low", 0.3))
            high = float(dist_kwargs.get("high", 3.0))
            g = torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
            sign = torch.randint(0, 2, shape, device=device).to(dtype) * 2 - 1
            return g * sign

        if distribution in ("normal", "mixed"):
            def _rand_norm():
                mu_low = float(dist_kwargs.get("mu_low", -5.0))
                mu_high = float(dist_kwargs.get("mu_high", 5.0))
                sigma_low = float(dist_kwargs.get("sigma_low", 0.1))
                sigma_high = float(dist_kwargs.get("sigma_high", 2.0))
                # Enforce the new framework standard for mixed/normal floats:
                # mu in [-5, 5], sigma in [0.1, 2].
                mu_low = max(-5.0, min(5.0, mu_low))
                mu_high = max(-5.0, min(5.0, mu_high))
                sigma_low = max(0.1, min(2.0, sigma_low))
                sigma_high = max(0.1, min(2.0, sigma_high))
                if mu_low > mu_high:
                    mu_low, mu_high = mu_high, mu_low
                if sigma_low > sigma_high:
                    sigma_low, sigma_high = sigma_high, sigma_low
                # Generate scalar parameters on CPU to avoid NPU scalar
                # synchronization (e.g. vector core exception 507035).
                mu = float(torch.empty(1).uniform_(mu_low, mu_high).item())
                sigma = float(torch.empty(1).uniform_(sigma_low, sigma_high).item())
                # NPU does not support torch.normal for bfloat16, so generate in
                # float32 and cast down to the target dtype.
                if dtype is torch.bfloat16:
                    return torch.normal(mu, sigma, shape, dtype=torch.float32, device=device).to(dtype)
                return torch.normal(mu, sigma, shape, dtype=dtype, device=device)

            if distribution == "mixed":
                # Legacy framework alternates the entire case between normal and
                # uniform, so multiple tensors in the same workload share the same
                # distribution family.  Use a deterministic per-case coin flip.
                # ctx.case_index is passed by gen_inputs; fall back to random when
                # called directly without a context.
                ctx = dist_kwargs.get("_ctx")
                if ctx is not None:
                    use_normal = ctx.case_index % 2 == 0
                else:
                    use_normal = random.random() < 0.5
                if use_normal:
                    return _rand_norm()
                else:
                    low = float(dist_kwargs.get("low", -5.0))
                    high = float(dist_kwargs.get("high", 5.0))
                    low = max(-5.0, min(5.0, low))
                    high = max(-5.0, min(5.0, high))
                    if low > high:
                        low, high = high, low
                    return torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)

        # randn 和任何未识别的 distribution 默认走标准正态
        return torch.randn(shape, dtype=dtype, device=device)

    if is_lp_float:
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)

    # Boolean
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # Integers
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        if range_spec is not None:
            low, high = range_spec
        else:
            ranges: dict[torch.dtype, tuple[int, int]] = {
                torch.int8: (-128, 128),
                torch.int16: (-1024, 1024),
                torch.int32: (-1024, 1024),
                torch.int64: (-1024, 1024),
            }
            low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


def load_safetensors(
    definition: Definition,
    workload: Workload,
) -> dict[str, torch.Tensor]:
    """Load safetensors inputs for a workload.

    Iterates workload.inputs, loads each SafetensorsInput file, validates
    shape and dtype against the definition, and returns CPU tensors.

    Args:
        definition: Kernel definition with input tensor specs.
        workload: Workload with InputSpec entries.

    Returns:
        Dict mapping input name to CPU tensor (pinned memory when available).

    Raises:
        RuntimeError: If safetensors is not installed.
        ValueError: If shape or dtype validation fails.
    """
    try:
        import safetensors.torch as st
    except ImportError as e:
        raise RuntimeError(
            "safetensors is required for SafetensorsInput workloads. "
            "Install with: pip install safetensors"
        ) from e

    expected_shapes = definition.get_input_shapes(workload.axes)
    safe_tensors: dict[str, torch.Tensor] = {}
    loaded_files: dict[str, dict[str, torch.Tensor]] = {}

    for name, input_spec in workload.inputs.items():
        if input_spec.type != "safetensors":
            continue

        path = str(input_spec.path)

        # Cache loaded files to avoid re-reading the same path
        if path not in loaded_files:
            loaded_files[path] = st.load_file(path)
        tensors = loaded_files[path]

        if input_spec.tensor_key not in tensors:
            raise ValueError(
                f"Missing key '{input_spec.tensor_key}' in '{path}'"
            )

        t = tensors[input_spec.tensor_key]

        # Validate shape
        expected_shape = expected_shapes[name]
        if list(t.shape) != expected_shape:
            raise ValueError(
                f"'{name}' expected shape {expected_shape}, got {list(t.shape)}"
            )

        # Validate dtype
        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(
                f"'{name}' expected dtype {expect_dtype}, got {t.dtype}"
            )

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()

        safe_tensors[name] = t

    return safe_tensors


def gen_inputs(
        definition: Definition,
        workload: Workload,
        device: str,
        seed: int,
        *,
        safe_tensors: Optional[dict[str, torch.Tensor]] = None,
        custom_inputs_fn: Optional[Callable] = None,
        case_index: int = 0,
) -> list[Any]:
    """Generate input values for a workload in definition order.

    Dispatches to the ``InputGeneratorRegistry`` so per-operator logic lives
    in dedicated generator classes rather than hard-coded branches here.

    Args:
        definition: Kernel definition with input tensor specs.
        workload: Workload with concrete axis values and InputSpec entries.
        device: Target device string (e.g., "npu:0").
        seed: RNG seed for reproducibility.
        safe_tensors: Pre-loaded safetensors dict from load_safetensors().
        custom_inputs_fn: Optional callable that returns dict[name, Tensor]
            for CustomInput entries. Signature:
            (resolved_axes: dict, device: torch.device) -> dict[str, Tensor].
        case_index: 0-based index of this workload within the evaluation batch.
            Used to match legacy framework's per-case distribution alternation
            (e.g. idx % 2 selects normal vs uniform).

    Returns:
        List of input values (torch.Tensor or scalar) in definition.inputs order.
    """
    from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
    from npu_kernelbench.evaluator.custominputs.context import InputGenerationContext

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

    ctx = InputGenerationContext(
        definition=definition,
        workload=workload,
        device=torch.device(device),
        seed=seed,
        safe_tensors=safe_tensors or {},
        custom_inputs_fn=custom_inputs_fn,
        case_index=case_index,
    )
    generator = get_input_generator_registry().resolve(definition)
    return generator.generate(ctx)

def get_workload_mode(
    definition: Definition,
    workload: Workload,
    override: Optional[str] = None,
) -> str:
    """Determine the evaluation mode for a workload.
        ``override`` takes precedence when provided (via CLI ``--mode``).
        None means full benchmark mode.
        """

    if override is not None:
        return override
    return "full_benchmark"

# def calculate_speedup(perf_baseline: PerfBaseline, solution_latency_ms: float) -> float:
#     """计算性能加速比。
#     speedup = perf_baseline.latency_ms / solution_latency_ms
#     """
#     if solution_latency_ms <= 0:
#         raise ValueError(f"solution_latency_ms must be > 0, got {solution_latency_ms}")
#     return perf_baseline.latency_ms / solution_latency_ms