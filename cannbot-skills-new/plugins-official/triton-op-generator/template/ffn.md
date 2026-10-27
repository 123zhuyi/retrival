---
name: ffn
description: FFN / MoE FFN / quantized FFN operators backed by torch_npu.npu_ffn or aclnnFFNV2. Covers dense FFN, GLU activations, int8 quantized FFN, and expert-token segmented MoE FFN.
metadata:
  type: reference
---

# FFN 算子模板

本文档记录 `torch_npu.npu_ffn` / `aclnnFFNV2` 类 FFN 算子的 Triton Ascend 生成约束。它是 FFN 专用模板；不要把这里的黑盒语义推广到所有 transformer、linear 或 quantization 算子。

适用触发特征：

- PyTorch baseline 调用 `torch_npu.npu_ffn(...)`。
- `forward` 参数包含 `x, weight1, weight2, activation`，并可能包含 `expert_tokens`, `expert_tokens_index`, `bias1`, `bias2`, `scale`, `offset`, `deq_scale1`, `deq_scale2`, `antiquant_scale1`, `antiquant_scale2`, `antiquant_offset1`, `antiquant_offset2`, `inner_precise`, `output_dtype`。
- case JSON/JSONL 中同时覆盖普通 dense FFN、GLU、int8 quant、MoE 分组等形态。

## §0 适用范围与分类

| 子类 | 判定特征 | 语义重点 |
|---|---|---|
| Dense FFN | 无 `expert_tokens`，无 `scale/deq_scale*` | `x @ weight1 -> activation -> @ weight2` |
| GLU FFN | `activation in {geglu, swiglu, reglu}` | 第一段 matmul 输出按最后一维二分；`activation(second_half) * first_half` |
| Quantized FFN | `x/weight1/weight2` 为 int8，存在 `scale` 或 `deq_scale1` | int8 dot、dequant、activation、requant、int8 dot、dequant |
| MoE FFN | 存在 `expert_tokens` | 每个 expert 处理连续 token 段，写回全局 token offset |

## §1 通用经验

### F1 语义权威是 `torch_npu.npu_ffn`

- **必须**以冻结后的 `Model.forward()` 实际调用为准；当它调用 `torch_npu.npu_ffn` 时，`torch_npu.npu_ffn` / `aclnnFFNV2` 是唯一精度权威。
- **禁止**把 docstring 中的 Python 公式当成最终语义；docstring 只能帮助理解参数含义。
- **Why:** `aclnnFFNV2` 存在未在 Python 公式中表达的实现细节，例如 GLU split 顺序、per-channel `deq_scale1` 索引重映射、量化前 fp16 算术链。

### F2 保留完整输入域

- **必须**保留全部 JSONL cases，不能为了先通过而裁剪掉量化、GLU 或 MoE cases。
- **必须**保持 `ModelNew.forward()` 参数顺序和可选参数行为与 `Model.forward()` 一致。
- **禁止**修改 frozen task、case JSON、verify tolerance 或通过 fallback 调用 `torch_npu.npu_ffn`。

### F3 Host 侧只做形状、分派和分配

- **允许**在 host 侧做 `contiguous()`、reshape、输出/中间 tensor 分配、activation id 映射、dtype/mode 分派、expert token offset 计算。
- **禁止**在 host 侧用 `torch.matmul`、`torch_npu.npu_ffn`、PyTorch activation 或 CPU fallback 承担核心计算。

## §2 Layer 1 设计约束

### F-L1.1 Dense FFN 使用两阶段 matmul

- **必须**用真实 `@triton.jit` kernel 完成两次矩阵乘：
  - stage1: `x @ weight1 + bias1 -> activation`
  - stage2: `hidden @ weight2 + bias2 -> output`
- **必须**在低精度输入上使用 fp32 accumulator。
- **可以**将 activation 融入 stage1，但 stage1 结果的 dtype/cast 时机必须跟目标语义保持一致。

### F-L1.2 GLU split 顺序使用 NPU 语义

对 `geglu`、`swiglu`、`reglu`，第一段 matmul 输出按最后一维二分：

```python
first_half = out1[..., :hidden]
second_half = out1[..., hidden:]
```

`torch_npu.npu_ffn` / `aclnnFFNV2` 对齐的顺序是：

```python
hidden = activation(second_half) * first_half
```

禁止生成：

```python
hidden = activation(first_half) * second_half
```

### F-L1.3 MoE token 段必须用全局 row offset

`expert_tokens` 表示每个 expert 的连续 token 数。生成 MoE 路径时：

- **必须**维护 `token_start`。
- stage1 临时 tensor 写入使用 `tmp_base = token_start * tmp.stride(0)`。
- stage2 输出写入使用 `out_base = token_start * out.stride(0)`。
- expert 权重使用 `expert_id * weight.stride(0)` 作为 expert base。
- 如果 frozen baseline 没有使用 `expert_tokens_index`，生成代码也不要自行引入重排语义。

常见错误：

```python
# 错误：每个 expert 都从局部 row 0 写 output，覆盖前面 expert 的结果。
out[row, col] = ...
```

应按全局 token offset 写回：

```python
# 伪代码
global_row = token_start + local_row
store(out + global_row * stride_om + col * stride_on, value)
```

### F-L1.4 Int8 quant FFN stage1 保留 fp32 中间量

int8 quant FFN 的语义链是：

```text
int8 x int8 -> int32 accumulator -> fp32 dequant/activation -> quantize to int8 -> int8 x int8 -> dequant -> output
```

生成代码应显式区分两个 stage1 中间：

```python
raw = torch.empty((M, N1), device=x.device, dtype=torch.float32)
q = torch.empty((M, N1), device=x.device, dtype=torch.int8)
```

其中 `raw` 保存 stage1 dequant + activation 后的 fp32 中间语义，`q` 保存进入第二段 int8 matmul 的量化结果。不要用 bf16 中间 tensor 替代 `raw`。

### F-L1.5 Per-channel `deq_scale1` 使用 768-column remap

当 int8 quant FFN 的 `scale` 是 per-channel，即 `scale.numel() > 1` 时，stage1 读取 `deq_scale1` 需要按 768 列消费模式重映射：

```python
deq_idx = col
if REMAP_DEQ:
    deq_idx = deq_idx - 256 * ((deq_idx // 256) % 3)
deq = load(deq_scale1 + deq_idx)
```

约束：

- 仅用于 stage1 `deq_scale1`。
- 仅在 `scale.numel() > 1` 的 per-channel quant 路径启用。
- stage2 `deq_scale2` 不做此 remap。
- scalar `scale` 不做此 remap。

### F-L1.6 Quantization arithmetic 使用 fp16 链

activation 后的量化步骤需要先按 fp16 做乘加，再转 fp32 做 `nearbyint`：

```python
v = val.to(fp16) * scale.to(fp16)
v = v + offset.to(fp16)
q = nearbyint(v.to(fp32))
q = clamp(q, -128, 127).to(int8)
```

不要直接用纯 fp32：

```python
q = nearbyint(val * scale + offset)
```

也不要把这个规则泛化到非 FFN 量化算子。

## §3 Layer 2 算法骨架

### F-L2.1 Host 分派

```python
def forward(x, weight1, weight2, activation, expert_tokens=None, ..., scale=None, offset=None, deq_scale1=None, deq_scale2=None, output_dtype=None):
    x2d = flatten_to_2d(x)

    if expert_tokens is not None:
        return run_moe_segments(...)

    if scale is not None or deq_scale1 is not None:
        return run_quant_ffn(...)

    if activation in {\"geglu\", \"swiglu\", \"reglu\"}:
        return run_glu_ffn(...)

    return run_dense_ffn(...)
```

### F-L2.2 Dense / GLU path

```python
# stage1
mm1 = dot(x, weight1) + optional_bias1

if activation is GLU:
    first = mm1[:, :hidden]
    second = mm1[:, hidden:]
    hidden = activation(second) * first
else:
    hidden = activation(mm1)

# stage2
out = dot(hidden, weight2) + optional_bias2
```

### F-L2.3 Quant path

```python
# stage1 quant
acc1_i32 = dot_i8_i8(x, weight1) + optional_bias1

deq_idx = col
if scale is per-channel:
    deq_idx = col - 256 * ((col // 256) % 3)

val_fp32 = acc1_i32.to(fp32) * deq_scale1[deq_idx].to(fp32)
act_fp32 = activation(val_fp32)

v_fp16 = act_fp32.to(fp16) * scale.to(fp16)
v_fp16 = v_fp16 + offset.to(fp16)
q_i8 = nearbyint(v_fp16.to(fp32))
q_i8 = clamp(q_i8, -128, 127).to(int8)

# stage2 quant
acc2_i32 = dot_i8_i8(q_i8, weight2) + optional_bias2
out = acc2_i32.to(fp32) * deq_scale2.to(fp32)
out = out.to(output_dtype or fp16)
```

## §4 Failure Routing For Mode C

| Failure pattern | First check | Repair direction |
|---|---|---|
| Only GLU cases fail with small relative error | Split order and intermediate cast placement | Use `activation(second_half) * first_half`; avoid unnecessary fp16 round-trip unless full verify proves it |
| MoE cases fail or show row blocks overwritten | token offset and expert base offsets | Use `token_start` for tmp/output base, and `expert_id * stride(0)` for weights/bias |
| Quant per-channel case has very low matched ratio | `deq_scale1` indexing | Apply 768/256 remap only to stage1 and only when `scale.numel() > 1` |
| Quant scalar case has high matched ratio but max cap failures | quant arithmetic/cast chain | Use fp16 multiply/add before `nearbyint`; keep stage2 deq unmapped |

## §5 Evidence From FFN Run

The fresh run under `ffn_gitcode_run_20260908_1709_9859` showed this progression:

| Candidate | Verify | Key observation |
|---|---:|---|
| `iter_1` | 38/53 | Initial generated implementation; MoE and quant/GLU failures |
| `iter_2` | 49/53 | MoE offset fix removed most failures |
| `iter_4_glu_swap` | 51/53 | GLU split order fixed cases 6 and 7 |
| `iter_7_quant_semantic` | 53/53 | Quant remap + fp32 raw + fp16 quant arithmetic fixed cases 26 and 27 |

This evidence should be treated as an FFN-specific precision rule set. It does not imply that every quantization operator uses the same `deq_scale1` remap or fp16 quant arithmetic.
