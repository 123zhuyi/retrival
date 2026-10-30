# Anti-cheat / Anti-hack Switches

npu-kernelbench includes a two-layer anti-cheat pipeline to detect solutions that bypass the intended kernel or game the benchmark.

## Defaults

By default, two checks are enabled:

- `ast_scan`
- `kernel_signature`

All other checks are opt-in. You can change the enabled set in three ways:

1. **CLI flags** (per-command override):
   ```bash
   uv run npu-kernelbench eval tests/samples/matmul_triton \
       --solution solution.json \
       --enable-anti-hack ast_scan \
       --enable-anti-hack kernel_signature
   ```

2. **`config.json`** (project-level default):
   ```json
   {
     "enable_anti_hack": ["ast_scan", "kernel_signature"]
   }
   ```
   To disable **all** checks, set the array to empty:
   ```json
   {
     "enable_anti_hack": []
   }
   ```

3. **`--mode correctness_only`** skips performance profiling **and** all anti-hack checks entirely.

## Available checks

### `ast_scan` (L1 static check)

Scans every `.py` source file in `sources` using the `HackDetector` AST visitor (`src/npu_kernelbench/evaluator/anti_hack.py`).

What it blocks:

- Forbidden imports: `vllm`, `cupy`, `cublas`, `cudnn`, `cufft`, `ctypes`, `torch.ops.aten`, `torch.ops.vllm`.
- Dynamic loading: `ctypes.CDLL`, `ctypes.cdll.LoadLibrary`, `torch.ops.load_library`.
- Code execution: `exec()`, `eval()`.
- Dynamic imports: `__import__(...)`, `importlib.import_module(...)` of forbidden modules.
- Forbidden `torch.*` calls outside the allow-list (e.g. `torch.mm`, `torch.matmul`, `torch.sum` are blocked unless explicitly whitelisted).
- Forbidden `torch_npu.*` calls outside the allow-list.
- Alias tricks (`import torch as th` + `th.matmul`).
- `getattr(torch, "matmul")` bypass attempts.

What is allowed (examples):

- Tensor creation: `torch.empty`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.randint`, `torch.full`, `torch.empty_like`, etc.
- Tensor manipulation: `torch.stack`, `torch.cat`, `torch.unsqueeze`, `torch.reshape`, `torch.view`, `torch.permute`, `torch.transpose`, `torch.contiguous`.
- Dtype/device helpers: `torch.float16`, `torch.bfloat16`, `torch.cuda.synchronize`, `torch.npu.synchronize`, etc.
- Context managers: `torch.no_grad`, `torch.manual_seed`, `torch.set_grad_enabled`.
- `torch_npu.profiler.*`, `torch_npu.npu.*`, `torch_npu.npu_*` direct op calls.

When to enable: almost always. This is the first line of defense.

### `kernel_signature` (L2 runtime check)

Verifies that the solution actually executes a declared kernel.

- For Python DSLs (`python`, `pytorch`, `triton`, `tilelang`): extracts function names that have decorators (e.g. `@triton.jit`).
- For AscendC / Catlass: extracts functions qualified with `__global__ __aicore__` or `__global__ __vector__`.
- At runtime, the evaluator checks whether at least one declared kernel appears in the profiler output (`kernel_details.csv`).

If the solution claims to implement a custom kernel but silently falls back to PyTorch, this check flags it.

You can help the checker by setting `declared_kernel` in `solution.json`.

### `output_caching`

Runs the solution 5 times with different random seeds. If the outputs are identical across runs, the solution may be returning a cached result.

- Integer/index outputs (e.g. `topk` indices) can legitimately collide, so the threshold is relaxed for all-integer outputs.
- Otherwise, fewer unique outputs than runs is treated as high-severity cheating.

### `monkey_patch`

Re-hashes the reference implementation code at runtime and compares it to `definition.reference_hash`. If they differ, the solution (or its environment) has modified the reference, which is a high-severity reward hack.

This only works when the `definition.json` includes a pre-computed `reference_hash`.

### `clock_anomaly`

Checks the coefficient of variation (CV = std / mean) of per-iteration durations recorded during profiling.

- Requires at least 10 timing samples.
- CV above `1.0` flags possible clock manipulation or unstable timing.
- Severity is medium because high CV can also come from legitimate system noise.

## Trace status

When an anti-hack check detects a problem, the trace status becomes `REWARD_HACK` and the `evaluation.log` field contains the violation details.

## Quick recipe: run full anti-cheat

```bash
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution tests/samples/matmul_triton/solution.json \
    --enable-anti-hack ast_scan \
    --enable-anti-hack kernel_signature \
    --enable-anti-hack output_caching \
    --enable-anti-hack monkey_patch \
    --enable-anti-hack clock_anomaly
```

## Quick recipe: disable everything

```bash
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution tests/samples/matmul_triton/solution.json \
    --mode correctness_only
```

Or place this `config.json` next to the problem:

```json
{
  "enable_anti_hack": []
}
```
