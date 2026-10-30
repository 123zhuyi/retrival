"""Anti-hack detection for NPU kernel evaluation.

Two-layer defense:
  L1 — Static AST scan of solution source files (whitelist/blacklist).
  L2 — Kernel signature check via profiler CSV (runtime).

Reward hack detection:
  - Output caching detection
  - Monkey patch detection (hash verification)
  - Clock anomaly detection (coefficient of variation)
  - Kernel signature mismatch (directory-level, all declared kernels)
"""

from __future__ import annotations

import ast
import csv
import hashlib
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import torch


# ============================================================
# L1: AST static scan — whitelists and blacklists
# ============================================================

ALLOWED_TORCH_API: set[str] = {
    # Tensor creation
    "torch.empty",
    "torch.zeros",
    "torch.ones",
    "torch.randn",
    "torch.rand",
    "torch.randint",
    "torch.arange",
    "torch.linspace",
    "torch.logspace",
    "torch.tensor",
    "torch.as_tensor",
    "torch.from_numpy",
    "torch.empty_like",
    "torch.zeros_like",
    "torch.ones_like",
    "torch.randn_like",
    "torch.full",
    "torch.full_like",
    # Tensor manipulation
    "torch.stack",
    "torch.cat",
    "torch.unsqueeze",
    "torch.squeeze",
    "torch.reshape",
    "torch.view",
    "torch.permute",
    "torch.transpose",
    "torch.contiguous",
    "torch.to",
    "torch.type",
    "torch.float",
    "torch.half",
    "torch.bfloat16",
    "torch.int",
    "torch.long",
    # Dtype
    "torch.float16",
    "torch.bfloat16",
    "torch.float32",
    "torch.float64",
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.bool",
    "torch.Size",
    # Constants
    "torch.pi",
    "torch.inf",
    "torch.nan",
    "torch.e",
    # Device / CUDA (NPU environment may map torch.cuda.*)
    "torch.cuda.synchronize",
    "torch.cuda.current_device",
    "torch.cuda.is_available",
    "torch.cuda.device_count",
    "torch.cuda.device",
    "torch.cuda.get_device_name",
    # Device / NPU (torch.npu is torch_npu injected into torch namespace)
    "torch.npu.synchronize",
    "torch.npu.current_device",
    "torch.npu.is_available",
    "torch.npu.device_count",
    "torch.npu.set_device",
    "torch.npu.get_device_name",
    # Context
    "torch.no_grad",
    "torch.manual_seed",
    "torch.set_grad_enabled",
    "torch.set_default_device",
    "torch.set_default_dtype",
}

ALLOWED_TORCH_NPU_API: set[str] = {
    "torch_npu.synchronize",
    "torch_npu.set_device",
    "torch_npu.get_device_name",
    "torch_npu.get_device_capability",
    "torch_npu.is_available",
    "torch_npu.device_count",
    # Prefix: all profiler APIs
    "torch_npu.profiler.",
    # Prefix: torch_npu.npu.Event etc.
    "torch_npu.npu.",
    # Prefix: torch_npu.npu_* direct op calls (e.g., npu_rotary_mul)
    "torch_npu.npu_",
}

# Backend-agnostic forbidden imports.
FORBIDDEN_IMPORTS: list[str] = [
    "vllm",
    "cupy",
    "cublas",
    "cudnn",
    "cufft",
    "ctypes",
    "torch.ops.aten",
    "torch.ops.vllm",
]


class HackDetector(ast.NodeVisitor):
    """AST visitor that detects hack patterns in solution source code.

    Checks:
    1. Torch API whitelist — any torch.X() not in ALLOWED_TORCH_API is flagged.
    2. torch_npu API whitelist — any torch_npu.X() not in ALLOWED_TORCH_NPU_API
       is flagged.
    3. Forbidden imports — import of blacklisted modules (ctypes, vllm, etc.),
       including __import__() and importlib.import_module().
    4. Dynamic loading — ctypes.CDLL / cdll.LoadLibrary / torch.ops.load_library.
    5. Code execution — exec() / eval().
    6. Alias tracking — import torch as th; th.matmul(...) is detected.
    7. getattr bypass — getattr(torch, "matmul") is detected.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []
        self._aliases: dict[str, str] = {}

    # -- Imports ----------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if self._is_forbidden_module(alias.name):
                self.violations.append(
                    f"Forbidden import: 'import {alias.name}' (line {node.lineno})"
                )
            if alias.asname:
                self._aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and self._is_forbidden_module(node.module):
            names = ", ".join(a.name for a in node.names)
            self.violations.append(
                f"Forbidden import: 'from {node.module} import {names}' "
                f"(line {node.lineno})"
            )
        if node.module:
            for alias in node.names:
                key = alias.asname or alias.name
                self._aliases[key] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    # -- Calls ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        call_chain = self._get_attr_chain(node.func)
        if call_chain:
            call_chain = self._resolve_alias(call_chain)

            # Forbidden torch API — skip torch.ops.* (custom op calls, e.g.,
            # AscendC's torch.ops.cpp_extension_acs.*; torch.ops.aten and
            # torch.ops.vllm are handled by FORBIDDEN_IMPORTS; torch.ops.
            # load_library has its own dynamic-loading check)
            if (
                self._is_torch_api(call_chain)
                and not self._is_torch_allowed(call_chain)
                and call_chain != "torch.ops.load_library"
                and not call_chain.startswith("torch.ops.")
            ):
                self.violations.append(
                    f"Forbidden torch API: '{call_chain}()' (line {node.lineno})"
                )

            # Forbidden torch_npu API
            if self._is_torchnpu_api(call_chain) and not self._is_torchnpu_allowed(
                call_chain
            ):
                self.violations.append(
                    f"Forbidden torch_npu API: '{call_chain}()' (line {node.lineno})"
                )

            # Forbidden module call — skip dynamic loading patterns (have own checks)
            if self._is_forbidden_module(call_chain) and not self._is_dynamic_loading(
                call_chain
            ):
                self.violations.append(
                    f"Forbidden call: '{call_chain}()' (line {node.lineno})"
                )

            # importlib.import_module("vllm") — via alias-resolved call chain
            if call_chain.endswith(".import_module"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    if self._is_forbidden_module(str(node.args[0].value)):
                        self.violations.append(
                            f"Forbidden dynamic import: "
                            f"'importlib.import_module(\"{node.args[0].value}\")' "
                            f"(line {node.lineno})"
                        )

        # getattr(torch, "matmul") bypass
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            base = self._get_attr_chain(node.args[0])
            if base:
                full = f"{base}.{node.args[1].value}"
                full = self._resolve_alias(full)
                if self._is_torch_api(full) and not self._is_torch_allowed(full):
                    self.violations.append(
                        f"Forbidden torch API via getattr: "
                        f"'getattr({base}, \"{node.args[1].value}\")' "
                        f"(line {node.lineno})"
                    )
                if self._is_torchnpu_api(full) and not self._is_torchnpu_allowed(full):
                    self.violations.append(
                        f"Forbidden torch_npu API via getattr: "
                        f"'getattr({base}, \"{node.args[1].value}\")' "
                        f"(line {node.lineno})"
                    )

        # __import__("vllm") — check both Name and Attribute forms
        import_name = None
        if isinstance(node.func, ast.Name):
            import_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            import_name = node.func.attr
        if import_name == "__import__":
            if node.args and isinstance(node.args[0], ast.Constant):
                if self._is_forbidden_module(str(node.args[0].value)):
                    self.violations.append(
                        f"Forbidden dynamic import: "
                        f"'__import__(\"{node.args[0].value}\")' (line {node.lineno})"
                    )

        # exec() / eval() — check both Name and Attribute forms
        if import_name in ("exec", "eval") and (isinstance(node.func, ast.Name) or call_chain in ("builtins.exec", "builtins.eval")):
            self.violations.append(
                f"Forbidden call: '{import_name}()' (line {node.lineno})"
            )

        # Dynamic loading: ctypes.CDLL / ctypes.cdll.LoadLibrary
        if self._is_dynamic_loading(call_chain):
            self.violations.append(
                f"Forbidden dynamic loading: '{call_chain}()' (line {node.lineno})"
            )

        # torch.ops.load_library
        if call_chain == "torch.ops.load_library":
            self.violations.append(
                f"Forbidden dynamic loading: '{call_chain}()' (line {node.lineno})"
            )

        self.generic_visit(node)

    # -- Attribute access -------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr_chain = self._get_attr_chain(node)
        if (
            attr_chain
            and self._is_forbidden_module(attr_chain)
            and not self._is_dynamic_loading(attr_chain)
        ):
            self.violations.append(
                f"Forbidden attribute access: '{attr_chain}' (line {node.lineno})"
            )
        self.generic_visit(node)

    # -- Helpers ----------------------------------------------------------

    def _is_forbidden_module(self, name: str) -> bool:
        for prefix in FORBIDDEN_IMPORTS:
            if name == prefix or name.startswith(prefix + "."):
                return True
        return False

    def _is_torch_api(self, call_chain: str) -> bool:
        return call_chain.startswith("torch.")

    def _is_torch_allowed(self, call_chain: str) -> bool:
        return call_chain in ALLOWED_TORCH_API

    def _is_torchnpu_api(self, call_chain: str) -> bool:
        return call_chain.startswith("torch_npu.")

    def _is_torchnpu_allowed(self, call_chain: str) -> bool:
        if call_chain in ALLOWED_TORCH_NPU_API:
            return True
        for entry in ALLOWED_TORCH_NPU_API:
            if entry.endswith((".", "_")) and call_chain.startswith(entry):
                return True
        return False

    def _is_dynamic_loading(self, call_chain: str | None) -> bool:
        if not call_chain:
            return False
        return call_chain in (
            "ctypes.CDLL",
            "ctypes.cdll.LoadLibrary",
        )

    def _resolve_alias(self, call_chain: str) -> str:
        if not call_chain:
            return call_chain
        parts = call_chain.split(".")
        first = parts[0]
        if first in self._aliases:
            parts[0] = self._aliases[first]
            return ".".join(parts)
        return call_chain

    @staticmethod
    def _get_attr_chain(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Call):
            return None
        else:
            return None
        return ".".join(reversed(parts))


# ============================================================
# L1: Public entry point
# ============================================================

def run_ast_scan(source_files: list[Path]) -> list[str]:
    """Scan Python source files for hack patterns.

    Args:
        source_files: List of absolute paths to .py source files.

    Returns:
        List of human-readable violation strings. Empty list if clean.
    """
    all_violations: list[str] = []
    for sf in source_files:
        if not sf.exists() or sf.suffix != ".py":
            continue
        try:
            code = sf.read_text()
        except Exception:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        detector = HackDetector()
        detector.visit(tree)
        if detector.violations:
            all_violations.append(f"[{sf.name}]")
            all_violations.extend(f"  {v}" for v in detector.violations)
    return all_violations


# ============================================================
# L2: Kernel signature check (all kernels in source directory)
# ============================================================

# AscendC kernel entry: __global__ [__aicore__|__vector__] ret_type name(...)
_ASCENDC_KERNEL_RE = re.compile(
    r"__global__\s+(?:__aicore__|__vector__)?\s*\w+\s+(\w+)\s*\("
)


def extract_kernel_signatures(source_dir: Path) -> set[str]:
    """Extract actual kernel function names from solution source files.

    - PythonDSL (.py): functions with decorators (e.g., @triton.jit).
    - AscendC (.cpp/.asc/.cc/.cu/.c): functions qualified with __global__.

    Args:
        source_dir: Directory containing solution source files.

    Returns:
        Set of kernel function names found in the sources.
    """
    signatures: set[str] = set()
    if not source_dir.exists():
        return signatures

    # PythonDSL — decorated functions
    for py_file in source_dir.glob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.decorator_list:
                signatures.add(node.name)

    # AscendC — __global__ functions
    for ext in ("*.cpp", "*.asc", "*.cc", "*.cu", "*.c"):
        for src_file in source_dir.glob(ext):
            try:
                text = src_file.read_text()
            except OSError:
                continue
            for m in _ASCENDC_KERNEL_RE.finditer(text):
                signatures.add(m.group(1))

    return signatures


def detect_kernel_signature_mismatch(
    source_dir: str | Path,
    trace_dir: str | Path,
) -> list[str]:
    """Compare recorded kernels against declared kernels in solution sources.

    The check is designed to catch solutions that bypass their declared custom
    kernels (e.g. by falling back to native PyTorch ops). It extracts every
    declared kernel from the solution sources (PythonDSL decorated functions
    and AscendC ``__global__`` functions) and verifies that at least one of
    them appears in the profiler's ``kernel_details.csv``. This avoids false
    positives when a source file declares helper kernels that are not launched
    for a particular workload.

    Args:
        source_dir: Directory containing solution source files.
        trace_dir: Directory containing profiler output (kernel_details.csv).

    Returns:
        List of human-readable mismatch strings. Empty list if at least one
        declared kernel was recorded, or if either directory is missing/empty.
    """
    if not source_dir or not trace_dir:
        return []

    src_path = Path(source_dir)
    trace_path = Path(trace_dir)
    if not src_path.exists() or not trace_path.exists():
        return []

    declared_kernels = extract_kernel_signatures(src_path)
    if not declared_kernels:
        return []

    kernel_names: set[str] = set()
    for csv_path in trace_path.glob("**/kernel_details.csv"):
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "")
                    if name:
                        kernel_names.add(name)
        except Exception:
            continue

    if not kernel_names:
        return []

    if any(
        declared in recorded or recorded in declared
        for declared in declared_kernels
        for recorded in kernel_names
    ):
        return []

    return [
        f"Kernel signature mismatch: none of the declared kernels {sorted(declared_kernels)} "
        f"appear in recorded kernels {sorted(kernel_names)}"
    ]


# ============================================================
# Reward hack detection checks
# ============================================================

@dataclass
class RewardHackResult:
    """Result of a single reward hack check."""

    detected: bool
    hack_type: str
    evidence: dict[str, Any] = field(default_factory=dict)
    severity: Literal["low", "medium", "high"] = "medium"


def _format_evidence(evidence: dict[str, Any]) -> str:
    """Flatten evidence dict into a short human-readable string."""
    parts = []
    for k, v in evidence.items():
        if isinstance(v, (list, tuple, set)):
            v = ", ".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def detect_output_caching(
    kernel_fn: Callable,
    input_generator: Callable[[int], tuple],
    *,
    num_runs: int = 5,
    base_seed: int = 200,
) -> RewardHackResult:
    """Detect output caching: run kernel N times with different inputs.

    If outputs are identical across runs with different inputs, the kernel
    may be caching and returning the same result.
    """
    outputs = []
    all_outputs_integer = []
    for i in range(num_runs):
        seed = base_seed + i
        inputs = input_generator(seed)
        with torch.no_grad():
            result = kernel_fn(*inputs)

        tensors = []
        tensor_is_integer = []

        def _collect(v):
            if isinstance(v, torch.Tensor):
                tensors.append(v.flatten().to(torch.float32))
                tensor_is_integer.append(v.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8))
            elif isinstance(v, (tuple, list)):
                for item in v:
                    _collect(item)
            else:
                tensors.append(torch.tensor(v).flatten().to(torch.float32))
                tensor_is_integer.append(False)

        _collect(result)
        if not tensors:
            outputs.append(torch.tensor([]))
        elif len(tensors) == 1:
            outputs.append(tensors[0])
        else:
            outputs.append(torch.cat(tensors))
        all_outputs_integer.append(all(tensor_is_integer) if tensor_is_integer else False)

    # Compare all pairs — count unique outputs
    unique_count = 0
    for i, out_i in enumerate(outputs):
        is_unique = True
        for j, out_j in enumerate(outputs):
            if i != j and torch.allclose(out_i, out_j, atol=1e-8):
                is_unique = False
                break
        if is_unique:
            unique_count += 1

    # Integer/index outputs (e.g. top-k indices) can legitimately collide across
    # seeds; only treat exact replication across all runs as caching.
    threshold = 1 if all(all_outputs_integer) else num_runs

    if unique_count < threshold:
        return RewardHackResult(
            detected=True,
            hack_type="output_caching",
            evidence={
                "num_runs": num_runs,
                "unique_outputs": unique_count,
                "ratio": unique_count / num_runs,
            },
            severity="high",
        )

    return RewardHackResult(
        detected=False,
        hack_type="output_caching",
        evidence={"num_runs": num_runs, "unique_outputs": unique_count},
    )


def detect_monkey_patch(
    reference_code: str,
    expected_hash: str,
) -> RewardHackResult:
    """Detect monkey patching: verify reference code hash matches expected."""
    if not expected_hash:
        return RewardHackResult(
            detected=False,
            hack_type="monkey_patch",
            evidence={"reason": "reference_hash empty, skipping"},
        )

    actual_hash = hashlib.sha256(reference_code.encode()).hexdigest()

    if actual_hash != expected_hash:
        return RewardHackResult(
            detected=True,
            hack_type="monkey_patch",
            evidence={
                "expected_hash": expected_hash[:16] + "...",
                "actual_hash": actual_hash[:16] + "...",
            },
            severity="high",
        )

    return RewardHackResult(
        detected=False,
        hack_type="monkey_patch",
        evidence={"hash": actual_hash[:16] + "..."},
    )


def detect_clock_anomaly(
    durations: list[float],
    *,
    max_cv: float = 1.0,
) -> RewardHackResult:
    """Detect abnormal NPU frequency via coefficient of variation in durations.

    High CV (>max_cv) across iterations may indicate dynamic clock manipulation.
    """
    if len(durations) < 10:
        return RewardHackResult(
            detected=False,
            hack_type="clock_anomaly",
            evidence={"reason": "insufficient samples (need >=10)", "count": len(durations)},
        )

    mean_d = statistics.mean(durations)
    if mean_d < 1e-9:
        cv = 0.0
    else:
        std_d = statistics.stdev(durations)
        cv = std_d / mean_d

    if cv > max_cv:
        return RewardHackResult(
            detected=True,
            hack_type="clock_anomaly",
            evidence={
                "cv": round(cv, 4),
                "threshold": max_cv,
                "mean_ms": round(mean_d, 4),
                "std_ms": round(statistics.stdev(durations), 4),
                "count": len(durations),
            },
            severity="medium",
        )

    return RewardHackResult(
        detected=False,
        hack_type="clock_anomaly",
        evidence={"cv": round(cv, 4), "mean_ms": round(mean_d, 4)},
    )


def detect_single_kernel_signature_mismatch(
    declared_kernel: str,
    trace_dir: str,
) -> RewardHackResult:
    """Detect kernel signature mismatch for a single declared kernel.

    Compares the declared kernel name against the names recorded in
    kernel_details.csv. Returns detected=True if the declared kernel
    does not appear in the recorded set.
    """
    from pathlib import Path
    import csv

    if not declared_kernel:
        return RewardHackResult(
            detected=False,
            hack_type="kernel_signature",
            evidence={"reason": "no declared kernel, skipping"},
        )

    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return RewardHackResult(
            detected=False,
            hack_type="kernel_signature",
            evidence={"reason": "trace directory not found"},
        )

    csv_files = list(trace_path.glob("**/kernel_details.csv"))
    if not csv_files:
        return RewardHackResult(
            detected=False,
            hack_type="kernel_signature",
            evidence={"reason": "no kernel_details.csv found"},
        )

    kernel_names: set[str] = set()
    for csv_path in csv_files:
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "")
                    if name:
                        kernel_names.add(name)
        except Exception:
            continue

    if not kernel_names:
        return RewardHackResult(
            detected=False,
            hack_type="kernel_signature",
            evidence={"reason": "no kernel names in csv"},
        )

    declared_kernel_str = str(declared_kernel)
    if not any(declared_kernel_str in name for name in kernel_names):
        return RewardHackResult(
            detected=True,
            hack_type="kernel_signature",
            evidence={
                "declared": declared_kernel_str,
                "actual": sorted(kernel_names),
            },
            severity="high",
        )

    return RewardHackResult(
        detected=False,
        hack_type="kernel_signature",
        evidence={"declared": declared_kernel_str, "matched": True},
    )


def run_all_checks(
    kernel_fn: Callable,
    input_generator: Callable[[int], tuple],
    reference_code: str,
    reference_hash: str,
    durations: list[float],
    trace_dir: str,
    enabled: Optional[list[str]] = None,
) -> list[RewardHackResult]:
    """Run enabled reward hack checks and return results.

    Note: ``kernel_signature`` is handled separately by the runner/evaluator
    because it needs the solution source directory, not a single kernel name.

    Args:
        enabled: List of check names to run. If None, runs all reward-hack
        checks except ``kernel_signature``.
    """
    if enabled is None:
        enabled = ["output_caching", "monkey_patch", "clock_anomaly"]
    enabled_set = set(enabled)

    results: list[RewardHackResult] = []

    if "output_caching" in enabled_set:
        results.append(detect_output_caching(kernel_fn, input_generator))
    if "monkey_patch" in enabled_set:
        results.append(detect_monkey_patch(reference_code, reference_hash))
    if "clock_anomaly" in enabled_set:
        results.append(detect_clock_anomaly(durations))

    return results