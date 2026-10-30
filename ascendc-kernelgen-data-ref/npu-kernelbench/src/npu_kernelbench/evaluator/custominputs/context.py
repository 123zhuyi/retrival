"""Generation context shared with custom input generators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.dtypes import DType, dtype_str_to_torch_dtype
from npu_kernelbench.data.workload import CustomInput, RandomInput, SafetensorsInput, ScalarInput, TensorListInput, Workload


@dataclass
class InputGenerationContext:
    """Context object passed to every custom input generator.

    Provides access to the definition, workload, device, RNG seed, and
    pre-loaded custom/safetensors inputs. It also exposes helper methods
    for resolving dtype/shape/scalars and generating random tensors so
    generators do not need to re-implement the generic logic.
    """

    definition: Definition
    workload: Workload
    device: torch.device
    seed: int
    safe_tensors: dict[str, torch.Tensor]
    custom_inputs_fn: Optional[Callable[[dict[str, int], torch.device], dict[str, Any]]]
    case_index: int = 0

    def resolve_dtype(self, name: str) -> torch.dtype:
        """Resolve the torch dtype for input ``name``.

        Prefers ``RandomInput.dtype`` or ``CustomInput.dtype`` from the workload,
        then the definition's tensor spec dtype. Raises ``ValueError`` if neither
        is available.
        """
        spec = self.definition.inputs[name]
        input_spec = self.workload.inputs.get(name)
        dtype_str = None
        if (
            isinstance(input_spec, RandomInput)
            and getattr(input_spec, "dtype", None) is not None
        ):
            dtype_str = input_spec.dtype
        elif (
            isinstance(input_spec, TensorListInput)
            and getattr(input_spec, "dtype", None) is not None
        ):
            dtype_str = input_spec.dtype
        elif (
            isinstance(input_spec, CustomInput)
            and getattr(input_spec, "dtype", None) is not None
        ):
            dtype_str = input_spec.dtype
        elif spec.dtype is not None:
            raw = spec.dtype
            dtype_str = raw.value if isinstance(raw, DType) else raw

        if dtype_str is None:
            raise ValueError(
                f"Cannot determine dtype for input '{name}': "
                f"no dtype in definition or workload"
            )
        return dtype_str_to_torch_dtype(dtype_str)

    def resolve_shape(self, name: str) -> tuple[int, ...]:
        """Resolve the concrete shape for input ``name``.

        Prefers ``RandomInput.shape`` or ``CustomInput.shape`` from the workload,
        then symbolic axis resolution from the definition. Raises ``ValueError``
        if neither is available.
        """
        spec = self.definition.inputs[name]
        input_spec = self.workload.inputs.get(name)
        shape = None
        if (
            isinstance(input_spec, RandomInput)
            and getattr(input_spec, "shape", None) is not None
        ):
            shape = tuple(input_spec.shape)
        elif (
            isinstance(input_spec, CustomInput)
            and getattr(input_spec, "shape", None) is not None
        ):
            shape = tuple(input_spec.shape)
        elif spec.shape is not None:
            shapes = self.definition.get_input_shapes(self.workload.axes)
            shape = shapes.get(name)
            if shape is None:
                raise ValueError(f"Cannot resolve shape for input '{name}'")
        else:
            raise ValueError(
                f"Cannot determine shape for input '{name}': "
                f"no shape in definition or workload"
            )
        return shape

    def scalar(self, name: str) -> Any:
        """Return the scalar value for input ``name`` if it is a ScalarInput."""
        input_spec = self.workload.inputs.get(name)
        if isinstance(input_spec, ScalarInput):
            return input_spec.value
        raise ValueError(f"Input '{name}' is not a ScalarInput")

    def derive_seed(self, *parts: str) -> int:
        """Return a deterministic integer seed derived from the base seed and parts.

        Python's built-in ``hash()`` is randomized across processes, so custom
        generators should use this helper when they need a stable per-name or
        per-workload seed.
        """
        payload = f"{self.seed}:{self.workload.uuid}:" + ":".join(parts)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return int(digest[:16], 16)

    def rand_tensor(
        self,
        name: str,
        shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
        **dist_kwargs: Any,
    ) -> torch.Tensor:
        """Generate a random tensor for input ``name`` using workload overrides.

        If ``shape`` or ``dtype`` are not provided, they are resolved from the
        workload/definition. Distribution parameters (``distribution``,
        ``low``, ``high``, ``range``, ``mu_low``, etc.) are read from the
        workload's ``RandomInput`` entry when present.
        """
        from npu_kernelbench.evaluator.generator import _rand_tensor

        if shape is None:
            shape = self.resolve_shape(name)
        if dtype is None:
            dtype = self.resolve_dtype(name)

        input_spec = self.workload.inputs.get(name)
        range_spec = None
        distribution = None
        if isinstance(input_spec, RandomInput):
            if getattr(input_spec, "range", None) is not None:
                range_spec = input_spec.range
            if getattr(input_spec, "distribution", None) is not None:
                distribution = input_spec.distribution
            for key in ("low", "high", "mu_low", "mu_high", "sigma_low", "sigma_high"):
                val = getattr(input_spec, key, None)
                if val is not None:
                    dist_kwargs.setdefault(key, float(val))

        # Allow callers to override the workload distribution explicitly
        # (e.g. custom generators that always want mixed).
        if "distribution" in dist_kwargs:
            distribution = dist_kwargs.pop("distribution")

        return _rand_tensor(list(shape), dtype, self.device, range_spec, distribution, _ctx=self, **dist_kwargs)

    def custom_tensor(self, name: str) -> Any:
        """Return the custom-generated tensor for input ``name``.

        Requires ``custom_inputs_fn`` to be set and to have produced a value
        for ``name``.
        """
        if self.custom_inputs_fn is None:
            raise RuntimeError(
                f"CustomInput for '{name}' requires custom_inputs_fn. "
                f"Definition '{self.definition.name}' has "
                f"custom_inputs_entrypoint='{self.definition.custom_inputs_entrypoint}'."
            )
        resolved_axes = self.definition.get_resolved_axes_values(self.workload.axes or {})
        custom_tensors = self.custom_inputs_fn(resolved_axes, self.device)
        if name not in custom_tensors:
            raise RuntimeError(f"CustomInput '{name}' not produced by custom_inputs_fn")
        val = custom_tensors[name]
        if isinstance(val, torch.Tensor):
            return val.to(device=self.device, non_blocking=True)
        return val

    def safetensor(self, name: str) -> torch.Tensor:
        """Return the pre-loaded safetensors input moved to the target device."""
        if name not in self.safe_tensors:
            raise RuntimeError(
                f"Missing required safetensors input '{name}'. "
                f"Call load_safetensors() before gen_inputs()."
            )
        return self.safe_tensors[name].to(device=self.device, non_blocking=True)
