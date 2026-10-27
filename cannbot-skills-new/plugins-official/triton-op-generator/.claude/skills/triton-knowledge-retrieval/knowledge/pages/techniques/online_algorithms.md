---
title: Online/Incremental Algorithms for Single-Pass Computation
category: techniques
tags: [online-softmax, flash-attention, welford, rmsnorm, logsumexp, numerical-stability]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Online/Incremental Algorithms for Single-Pass Computation

## Overview

Online (incremental) algorithms process data in blocks, maintaining running statistics rather than materializing the full intermediate result. On Ascend, this is critical for:
1. Keeping tile data within UB (192KB)
2. Eliminating HBM round-trips for intermediate tensors
3. Numerical stability (avoiding overflow in exp, sum)

## Online Softmax (Flash Attention Core)

The naive softmax over a full attention row requires `O(seq_len)` memory for the full attention matrix. Online softmax processes the attention scores block-by-block, maintaining a running max and sum.

### Algorithm

For each block of scores `scores_block`:

```
m_ij = max(m_i, max(scores_block))        # new running max
p = exp(scores_block - m_ij)              # numerically stable exp
l_ij = sum(p)                             # partial denominator
alpha = exp(m_i - m_ij)                   # correction factor for old running sum
l_i = l_i * alpha + l_ij                 # corrected running denominator
acc = acc * alpha + (p @ V_block)         # corrected running numerator (output)
m_i = m_ij                               # update running max
```

Final output: `output = acc / l_i`

### Memory Reduction

| Approach | Memory for Attention | Notes |
|----------|---------------------|-------|
| Standard softmax | O(S²) | Full N×N attention matrix |
| Online softmax | O(S) | Only running max, sum, acc vectors |

For S=4096, fp16: standard needs 4096² * 2B = 32MB. Online needs ~4096 * 3 * 4B = 48KB (fits in UB).

### Triton Implementation Pattern

```python
@triton.jit
def flash_attention_kernel(Q, K, V, Out, ...):
    # Initialize running state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Online softmax loop over K/V blocks
    for k_start in range(0, seq_len, BLOCK_N):
        # Load scores block
        k = tl.load(K_ptr + k_start * d + k_offs, mask=k_mask)
        scores = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]
        scores = scores * scale           # scale by 1/sqrt(d_k)

        # Online softmax update
        m_ij = tl.max(scores, axis=1)         # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)         # new running max
        p = tl.exp(scores - m_new[:, None])   # [BLOCK_M, BLOCK_N]
        l_ij = tl.sum(p, axis=1)              # [BLOCK_M]
        alpha = tl.exp(m_i - m_new)           # correction factor

        # Update accumulators
        l_i = l_i * alpha + l_ij
        v = tl.load(V_ptr + k_start * d + v_offs, mask=v_mask)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    # Final normalization
    output = acc / l_i[:, None]
    tl.store(Out_ptr + ..., output.to(tl.float16), ...)
```

## Welford Online Mean/Variance (LayerNorm Single-Pass)

The Welford algorithm computes mean and variance in a single pass without storing the full dataset. This is numerically more stable than the two-pass approach and avoids loading the row twice.

### Algorithm

```
count, mean, M2 = 0, 0.0, 0.0
for x in elements:
    count += 1
    delta = x - mean
    mean += delta / count
    M2 += delta * (x - mean)    # Note: uses updated mean
variance = M2 / count           # population variance
# or: M2 / (count - 1)         # sample variance
```

### Why It Is Numerically Stable

The Welford algorithm avoids computing `sum(x²) - n*mean²`, which suffers from catastrophic cancellation when mean is large relative to variance. The incremental delta formulation has much better floating-point stability.

### Triton Pattern for Block Welford

```python
@triton.jit
def layernorm_welford_kernel(X_ptr, W_ptr, B_ptr, Y_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    # Online Welford over column blocks
    count = 0
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    M2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in range(0, n_cols, BLOCK_SIZE):
        x = tl.load(X_ptr + pid * n_cols + col_start + offs,
                    mask=(col_start + offs) < n_cols).to(tl.float32)
        count += 1
        delta = x - mean
        mean = mean + delta / count
        M2 = M2 + delta * (x - mean)

    # Aggregate across the block (reduce to scalar)
    total_mean = tl.sum(mean) / (n_cols / BLOCK_SIZE)  # simplification
    variance = tl.sum(M2) / n_cols
    # ... apply normalization
```

## Online RMS (RMSNorm Single-Pass)

RMSNorm does not need mean subtraction, only RMS computation. This simplifies the online algorithm:

```
# Accumulate sum(x²) in fp32
sum_sq = 0.0
for block in row:
    x = load(block).to(fp32)
    sum_sq += sum(x * x)

rms = sqrt(sum_sq / N + eps)

# Second pass: normalize
for block in row:
    x = load(block).to(fp32)
    x_norm = x / rms * weight
    store(x_norm)
```

For large hidden dims that don't fit in UB, this two-pass approach is necessary. For hidden dims that fit in UB, the full row can be processed in one pass.

**Implementation note**: Always accumulate `sum_sq` in fp32, even if input is fp16/bf16. fp16 has insufficient precision for squaring and summing thousands of values.

```python
# In Triton
sum_sq = tl.zeros([1], dtype=tl.float32)
x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
sum_sq += tl.sum(x * x, axis=0)
rms = tl.sqrt(sum_sq / n_cols + eps)
```

## Log-Sum-Exp Trick (Numerically Stable Reduction)

For computing `log(sum(exp(x)))` (used in cross-entropy, softmax, log-probabilities):

**Naive (overflows for large x)**:
```
log_sum_exp = log(sum(exp(x)))    # exp(x) overflows for x > 88
```

**Stable (subtract max first)**:
```
m = max(x)
log_sum_exp = m + log(sum(exp(x - m)))    # exp(x - m) ≤ 1, never overflows
```

This is numerically equivalent because:
```
log(sum(exp(x))) = log(sum(exp(x - m) * exp(m)))
                 = log(exp(m) * sum(exp(x - m)))
                 = m + log(sum(exp(x - m)))
```

### Online Log-Sum-Exp

```
running_max = -inf
running_sum = 0.0
for block in x_blocks:
    block_max = max(block)
    new_max = max(running_max, block_max)
    running_sum = running_sum * exp(running_max - new_max) + sum(exp(block - new_max))
    running_max = new_max
log_sum_exp = running_max + log(running_sum)
```

This is the same correction-factor pattern as online softmax.

## Connections Between Algorithms

All four algorithms share the same structure:
1. Maintain a running statistic (max, mean, sum)
2. When new data arrives, apply a correction factor to the old statistic
3. After all blocks, finalize (divide by count, add log, etc.)

The correction factor is always `exp(old_max - new_max)` or `delta / count`. This pattern can be applied to any reduction that has a "merge" operation.

## 相关页面
- [ops/attention](../ops/attention.md) — Flash Attention implementation using online softmax
- [ops/normalization](../ops/normalization.md) — LayerNorm (Welford) and RMSNorm implementations
- [techniques/tiling](tiling.md) — Two-pass tiling for large hidden dims
- [techniques/data_types](data_types.md) — FP32 accumulation for numerical stability
- [workflows/profiling](../workflows/profiling.md) — Detecting numerical issues in profiling

## 被以下页面引用
- [ops/attention](../ops/attention.md) — Flash Attention 在线 Softmax（核心公式）
- [ops/normalization](../ops/normalization.md) — Welford 单遍 LayerNorm
- [ops/reduction](../ops/reduction.md) — log-sum-exp trick
- [experience_index](../experience_index.md) — exp_online_softmax_001, exp_online_mean_var_001
