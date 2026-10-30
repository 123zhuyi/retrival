# Level3 Benchmark Migration Example

This note uses `BatchMatmul` as the smallest level3 migration example.

## Mental Model

Old benchmark format:

```text
npu_benchmark/level3/1_BatchMatmul.py
npu_benchmark/level3/1_BatchMatmul.json
```

New kernelbench format:

```text
npu-kernelbench/data/kernel_generator/level3/BatchMatmul/definition.json
npu-kernelbench/data/kernel_generator/level3/BatchMatmul/workload.jsonl
```

In plain terms:

```text
old .py forward()        -> definition.json reference
old .py get_input_groups -> workload.jsonl plus default/custom input generator
old .json case lines     -> workload.jsonl case lines
```

## Step 1: Find the Math

In the old file, the actual operator behavior is the `forward` body:

```python
def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.bmm(A, B)
```

That becomes the new `definition.json` reference:

```json
"reference": "import torch\n\ndef run(A, B):\n    return torch.bmm(A, B)\n"
```

The important translation is `forward(...)` to a top-level `run(...)` function.

## Step 2: Name Inputs and Outputs

Old cases store inputs as a list:

```json
{"inputs": [
  {"name": "A", "type": "tensor", "dtype": "float32", "shape": [1, 8, 16]},
  {"name": "B", "type": "tensor", "dtype": "float32", "shape": [1, 16, 8]}
]}
```

New workloads store inputs by name:

```json
{"uuid": "BatchMatmul-001", "axes": null, "inputs": {
  "A": {"type": "random", "shape": [1, 8, 16], "dtype": "float32", "distribution": "normal", "mu_low": -100, "mu_high": 100, "sigma_low": 1, "sigma_high": 25},
  "B": {"type": "random", "shape": [1, 16, 8], "dtype": "float32", "distribution": "normal", "mu_low": -100, "mu_high": 100, "sigma_low": 1, "sigma_high": 25}
}}
```

For this operator:

```text
A shape: [batch, m, k]
B shape: [batch, k, n]
y shape: [batch, m, n]
```

The old level3 generators choose one distribution family for each case:

```text
case 1, 3, 5, ... -> normal(mu, sigma)
case 2, 4, 6, ... -> uniform(-5, 5)
```

The new workload records that choice explicitly for every floating-point
input. This keeps shape, dtype, and value distribution aligned with the old
benchmark.

## Step 3: Decide Whether a Custom Input Generator Is Needed

`BatchMatmul` does not need custom input logic. The workload fully describes
the shape, dtype, and distribution of `A` and `B`, and the new default
generator supports all three.

Use default generator when:

```text
inputs are independent random tensors or scalar attrs
```

Add a custom generator when:

```text
inputs must satisfy relationships, such as offsets, masks, indices, cache lengths, or optional None tensors
```

For example, `GroupedMatmul` needs a level3 custom generator because `offsets` must match the shape relationship between `A` and `B`.

## Step 4: Upgrade to GroupedMatmul

`GroupedMatmul` is the next level up from `BatchMatmul` because one input is not ordinary random data.

Latest old benchmark behavior:

```python
total_rows = A_info["shape"][0]
num_groups = B_info["shape"][0]
group_size = total_rows // num_groups
offsets = torch.tensor(
    [(i + 1) * group_size for i in range(num_groups)],
    dtype=torch.int32,
)
```

That means `offsets` stores the exclusive end row of every group:

```text
A shape:       [8, 16]
B shape:       [2, 16, 32]
offsets:       [4, 8]
group 0 uses:  A[0:4] @ B[0]
group 1 uses:  A[4:8] @ B[1]
```

So the new workload can still describe `offsets` as an int32 input, but the default random generator must not be used for its value. Random offsets would produce invalid or unstable groups.

The migration therefore adds:

```text
npu-kernelbench/src/npu_kernelbench/evaluator/custominputs/level3/grouped_matmul.py
```

Its job is small: generate random `A` and `B`, then derive legal `offsets` from their shapes.

## Checklist

For each level3 operator:

```text
1. Copy the math from old forward() into definition.reference as run().
2. Describe inputs and outputs in definition.json.
3. Convert every old JSON case into one workload.jsonl line.
4. Preserve each case's float distribution as well as shape and dtype.
5. Use "random" for tensor inputs and "scalar" for attrs.
6. Put relational inputs such as offsets in a custom input generator.
7. Validate definition.json and workload.jsonl with the kernelbench data models.
```
