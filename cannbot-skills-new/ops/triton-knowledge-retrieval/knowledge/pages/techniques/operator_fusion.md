---
title: Operator Fusion Strategies
category: techniques
tags: [fusion, gemm, activation, rmsnorm, rope, layernorm, quantization, hivm, constexpr]
hardware: [910b2, 910b3]
last_updated: 2026-07-06
---

# Operator Fusion Strategies

## Overview

Operator fusion eliminates intermediate HBM reads and writes between consecutive operations by combining them into a single kernel pass. For LLM inference and training, fused kernels routinely provide 25-50% speedup over separate kernels due to the high ratio of memory bandwidth to compute.

## Why Fuse?

Each separate operation requires:
1. Writing output to HBM (MTE3 store, ~1.8TB/s)
2. Reading that output back from HBM for the next op (MTE2 load, ~1.8TB/s)

For memory-bound ops (elementwise, normalization), the HBM round-trip is the dominant cost. Fusion keeps data in UB (192KB, on-chip, zero HBM cost) between the two operations.

**Example**: GEMM + RMSNorm separately:
```
GEMM: load A,B → compute → store C to HBM        # 1 HBM write
RMSNorm: load C from HBM → compute → store Y      # 1 HBM read + 1 HBM write
Total HBM traffic: 3 operations
```

**Fused**:
```
GEMM+RMSNorm: load A,B → GEMM → RMSNorm → store Y  # No intermediate HBM
Total HBM traffic: 1 operation
```

## Common Fusion Patterns and Speedups

### 1. GEMM + Activation (GELU/SiLU/ReLU)

Fuse post-accumulation activation into the matmul kernel. Apply activation to the `acc` tile before storing.

**Speedup**: 15-20%

```python
@triton.jit
def gemm_gelu_kernel(..., HAS_GELU: tl.constexpr):
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        acc += tl.dot(a, b)

    if HAS_GELU:
        # Fused activation: applied in UB before store
        acc = acc * tl.sigmoid(1.702 * acc)  # GELU approximation
    tl.store(C_ptr, acc.to(tl.float16), ...)
```

### 2. GEMM + Bias

Add bias vector to each row of the output tile in the same kernel:

**Speedup**: 10-15%

```python
if HAS_BIAS:
    bias = tl.load(bias_ptr + n_offs, mask=n_mask)
    acc = acc + bias[None, :]  # broadcast to [BLOCK_M, BLOCK_N]
```

### 3. GEMM + RMSNorm

Normalize the output tile immediately after GEMM accumulation. Eliminates the HBM round-trip for large output matrices.

**Speedup**: ~35%

```python
# After GEMM accumulation
rms = tl.sqrt(tl.sum(acc * acc, axis=1, keep_dims=True) / N + eps)
acc = acc / rms * weight[None, :]
tl.store(C_ptr, acc.to(tl.float16), ...)
```

### 4. GEMM + RoPE

Apply Rotary Position Embedding to Q or K projection output immediately after GEMM.

**Speedup**: ~25%

```python
# After GEMM: reshape [seq, d_head], apply RoPE
cos = tl.load(cos_ptr + pos_offs)
sin = tl.load(sin_ptr + pos_offs)
x1, x2 = acc[:, :d_half], acc[:, d_half:]
acc_rope = tl.join(x1 * cos - x2 * sin, x2 * cos + x1 * sin)
tl.store(Q_ptr, acc_rope.to(tl.float16), ...)
```

### 5. Skip + LayerNorm (Residual + Norm)

Compute `(X + residual)` and then `LayerNorm` in a single pass:

**Speedup**: ~30%

```python
x = tl.load(X_ptr + row * n_cols + offs, mask=mask)
residual = tl.load(R_ptr + row * n_cols + offs, mask=mask)
x = x + residual  # residual add in UB

# LayerNorm (single-pass Welford or two-pass)
mean = tl.sum(x, axis=0) / n_cols
var = tl.sum((x - mean) ** 2, axis=0) / n_cols
x_norm = (x - mean) / tl.sqrt(var + eps)
y = x_norm * weight + bias
tl.store(Y_ptr + ..., y, mask=mask)
```

### 6. RMSNorm + Quantization

Normalize then quantize in a single kernel. Eliminates separate quantization pass.

**Speedup**: ~50%

```python
# RMSNorm
rms = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=0) / n_cols + eps)
x_norm = x_fp32 / rms * weight

# Quantize immediately (still in UB)
scale = tl.max(tl.abs(x_norm), axis=0) / 127.0
x_quant = (x_norm / scale).to(tl.int8)
tl.store(Y_ptr, x_quant, mask=mask)
tl.store(scale_ptr, scale)
```

### 7. QKV RMSNorm + RoPE

Triple fusion for Q and K heads after QKV projection: normalize Q, normalize K, apply RoPE to both.

**Speedup**: ~25% (relative to doing RMSNorm and RoPE separately)

### 8. Conv1D + State Update + SiLU Activation（双缓冲状态融合）

将卷积计算、状态更新（滑动窗口移位）、SiLU 激活融合为单个 kernel。除消除中间 HBM 读写外，还使用**双缓冲状态更新**模式：

```python
# host 端：分配独立 update buffer
conv_state_update = torch.empty_like(conv_state)

# kernel 内：写入独立 buffer，多 program 并发无竞争
tl.store(conv_state_update_ptr + state_offset, new_state_slice)

# host 端：一次性同步（原子性保证）
conv_state.copy_(conv_state_update)
```

**为什么用双缓冲而非直接写入**：
- 多个 program 并发写入 `conv_state` 的不同区域时，直接写入可能触发伪共享
- 写入独立 buffer 后一次性 `copy_`，保证状态更新的原子性
- 读写分离消除了 WAR（Write-After-Read）依赖

**融合内容**：状态历史加载 → 拼接当前输入 → 卷积计算 → SiLU 激活 → 状态提取保存。原本需要 4 个独立操作，融合后全部在 UB 内完成。

**Speedup**: ~35%

经验：`exp_causal_conv1d_update_001`、`exp_causal_conv1d_update_007`；详见 [ops/convolution.md](../ops/convolution.md)

### 10. Merge Kernel 括号内累加融合

分块递推算法中，merge 步的公式 `X[i,j] = -X[i,i] · Σ A[i,k]·X[k,j]` 需要在乘法前先累加多个中间矩阵乘积。利用 Triton 的融合能力在同一 kernel 内完成累加+矩阵乘，消除中间结果的 HBM 写回：

```python
# 括号内先累加，再一次性做矩阵乘 → 中间结果在寄存器中
b_Ai_31 = -tl.dot(
    b_Ai_33,
    tl.dot(b_A_31, b_Ai_11) + tl.dot(b_A_32, b_Ai_21)  # 累加在寄存器中
)

# 三级累加（64×64 最远块）
b_Ai_41 = -tl.dot(
    b_Ai_44,
    tl.dot(b_A_41, b_Ai_11) + tl.dot(b_A_42, b_Ai_21) + tl.dot(b_A_43, b_Ai_31)
)
```

**融合内容**：多次 16×16 tl.dot 的中间结果在寄存器中累加后，一次性作为外层 tl.dot 的输入。原本需要写回 HBM 的中间矩阵乘积直接组合在表达式中，编译器将其融合为寄存器内操作。

**通用性**：适用于所有需要累加多个矩阵乘积的 kernel（分块求逆、分块 Cholesky、block-sparse attention 中的块间归约）。

经验：`exp_solve_tril_004`；详见 [ops/solve_tril.md](../ops/solve_tril.md)

### 9. Conv+Bias+SiLU 三合一融合（循环内融合 + 寄存器驻留）

将卷积计算、Bias 加法、SiLU 激活在单 kernel 的每次迭代内完成。**关键**：bias 向量在循环外一次性加载到寄存器，循环内每次迭代复用：

```python
# 循环外：bias 一次性加载（寄存器驻留）
if HAS_BIAS:
    bias = tl.load(bias_ptr + d_offs, mask=mask_d)

for i in range(seq_len):
    # 滑动窗口计算（寄存器内）
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3  # fp32

    if HAS_BIAS:
        acc = acc + bias                          # fp32 bias add（复用寄存器）
    if ACTIVATION == 'silu':
        acc = acc / (1 + tl.exp(-acc))            # fp32 SiLU

    tl.store(y_ptrs + i * stride_y, acc.to(OUT_DTYPE), mask=mask_d)
    # 寄存器移位...
```

**核心收益**：
1. 消除 2 个额外 kernel launch（延迟降低 30-40%）
2. Bias 寄存器驻留节省 `(seq_len - 1)` 次重复 load
3. fp32 SiLU 保证数值稳定性（避免 fp16 下 `exp(-x) → inf`）

**与 §8 Conv1D+State+SiLU 的关系**：§8 面向 Update kernel（短序列 + 状态持久化 + 双缓冲）；§9 面向 Prefill kernel（长序列 + 无状态 + 滑动窗口寄存器复用）。两者共同的融合原则：激活函数在 fp32 累加器上计算，constexpr 控制融合路径。

经验：`exp_causal_conv1d_prefill_006`；详见 [ops/convolution.md](../ops/convolution.md)

## Cube/Vector Heterogeneous Fusion

For kernels that contain both GEMM (Cube unit) and elementwise (Vector unit) operations, `dl.async_task` enables the two units to run in parallel:

```python
import triton_ascend.language as dl

@triton.jit
def fused_gemm_norm_async(...):
    # Cube computes GEMM tile i+1 while Vector normalizes tile i
    with dl.async_task(scope="cube"):
        acc = tl.dot(a_tile, b_tile)

    with dl.async_task(scope="vector"):
        rms = tl.sqrt(tl.sum(acc * acc, axis=1) / N + eps)
        out = (acc / rms[:, None]) * weight
        tl.store(Y_ptr + ..., out)
```

## HiVM Flags for Cube/Vector Balance

For complex fused kernels, HiVM flags control how the compiler balances Cube and Vector work:

| Flag | Purpose |
|------|---------|
| `enable_hivm_auto_cv_balance` | Automatically balance Cube/Vector workloads |
| `tile_mix_vector_loop` | Control Vector core split granularity |
| `tile_mix_cube_loop` | Control Cube core split granularity |

```python
@triton.jit
def fused_kernel(...):
    dl.compile_hint("enable_hivm_auto_cv_balance")
    dl.compile_hint("tile_mix_vector_loop=4")
    dl.compile_hint("tile_mix_cube_loop=2")
    # ... kernel body
```

## Compile-Time Specialization with `tl.constexpr`

Use `tl.constexpr` boolean flags to eliminate unused branches at compile time:

```python
@triton.jit
def flexible_gemm_kernel(
    ...,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_GELU: tl.constexpr,
):
    acc = compute_gemm(...)

    if HAS_BIAS:        # eliminated at compile time if False
        acc += bias
    if HAS_RESIDUAL:    # eliminated at compile time if False
        acc += residual
    if HAS_GELU:        # eliminated at compile time if False
        acc = gelu(acc)

    tl.store(C_ptr, acc, ...)
```

This avoids runtime branch prediction costs and enables the compiler to optimize code paths independently.

## Fusion Constraints

**Only fuse when**:
1. Operations are data-dependent: output of op A is the input of op B
2. The fused kernel fits the UB constraints (all intermediate tensors fit in 192KB)
3. The fused ops have compatible parallelism (same grid structure)

**Do NOT fuse**:
- Independent ops with different parallelism (e.g., separate batch dims)
- Ops where intermediate results are needed by multiple downstream consumers
- Ops whose combined tile size exceeds UB capacity
- Ops where one component has hard alignment constraints incompatible with the other（如 GEMM 的 BLOCK_N=256 vs GroupNorm 的 cpg=64 不可调和）

**双缓冲融合（Double-Buffered Fusion）**：当多个 program 并发写入同一共享状态（如 conv_state）时，使用独立 write buffer + 最终 `copy_` 同步，避免并发写入竞争。适用于状态持久化的融合模式（Conv1D+State、SSM chunk scan+state、KV cache update）。

## 相关页面
- [techniques/ilp](ilp.md) — dl.async_task for Cube/Vector overlap
- [techniques/tiling](tiling.md) — Ensuring fused tiles fit in UB
- [techniques/persistent_kernels](persistent_kernels.md) — Combining fusion with persistent pattern
- [ops/matmul](../ops/matmul.md) — GEMM + activation/bias/norm fusion examples
- [ops/attention](../ops/attention.md) — Flash Attention as a canonical fusion example
- [hardware/ascend_910b](../hardware/ascend_910b.md) — Cube/Vector architecture, HiVM flags

## 被以下页面引用
- [ops/matmul](../ops/matmul.md) — GEMM+Activation/Bias/RMSNorm/RoPE 融合变体
- [ops/elementwise](../ops/elementwise.md) — 激活函数内联进前序 matmul
- [ops/normalization](../ops/normalization.md) — Skip+LayerNorm、RMSNorm+Quant 融合
- [ops/attention](../ops/attention.md) — Flash Attention 作为融合典范
- [ops/convolution](../ops/convolution.md) — Conv1D+State+SiLU 双缓冲融合实例；Conv+Bias+SiLU 三合一融合实例
- [ops/solve_tril](../ops/solve_tril.md) — Merge kernel 括号内累加融合（多次 tl.dot 结果寄存器内归约）
- [ops/fused_ops](../ops/fused_ops.md) — 常见融合组合汇总
- [techniques/compute_and_branch](compute_and_branch.md) — DIM_IS_MULTIPLE / W_IS_4 constexpr 融合路径控制
- [techniques/ilp](ilp.md) — 滑动窗口寄存器复用与融合的组合使用
