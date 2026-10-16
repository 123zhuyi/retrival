---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: 归约算子优化
description: 归约算子优化
tags: [reduction, sum, max, min, logsumexp, split_k, parallel_reduction, numerical_stability, 910b2, 910b3, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/reduction.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/reduction.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/reduction.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# 归约算子优化

## 概述

归约算子沿指定维度将多个元素合并为更少的输出元素，典型操作包括 sum、max、min、argmax、logsumexp 等。在深度学习推理/训练中，归约广泛出现于 softmax、cross-entropy loss、LayerNorm、split-K matmul 等场景。

在 Ascend 910b2/910b3 上，归约算子优化的核心挑战为：

1. **数值精度**：fp16/bf16 累加易溢出，尤其对 logsumexp 等涉及 exp 的场景
2. **并行度**：大 K 维度归约（K ≥ 4096）需要跨核并行（split-K）充分利用全部 AI Core
3. **多遍开销**：在线算法可将两遍/多遍归约压缩为单遍，减少 HBM 带宽消耗
4. **联合归约**：argmax+max 等联合操作可在单遍中同时完成，避免重复扫描

## 检测标准

满足以下任意模式时，将算子归类为 reduction：

- Triton 原语：`tl.sum`、`tl.max`、`tl.min`、`tl.reduce`
- PyTorch 原语（含 `dim=` 参数）：`torch.sum(dim=)`、`torch.mean(dim=)`、`torch.max(dim=)`、`torch.logsumexp(dim=)`
- 输出张量维度少于输入张量（沿某维度折叠）
- split-K 并行归约模式（多核写入部分和，再全局合并）
- 间接归约：cross-entropy（含 log-softmax）、histogram（含原子 binning）、cumsum（前缀扫描）

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | 输入 fp16/bf16；累加必须 fp32 以避免溢出 |
| 归约维度大小 K | K ≥ 4096 时 split-K 收益显著；K < 256 时单核即可 |
| 并行维度大小 M | M 足够大时每行分配一个 program；M 小时用 persistent kernel |
| logsumexp 场景 | 必须使用 log-sum-exp trick（先减 max 再 exp），防止 exp 溢出 |
| split-K 场景 | 需要额外的全局 reduction buffer（fp32 累加），或使用两阶段方案 |

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 来源文件 |
|------|--------|-------|----------|----------|
| fp16/bf16 输入累加溢出 | exp_reduction_fp32_001 | 30% | fp32 累加（提升精度同时提升性能） | numerical_stability.md |
| logsumexp 数值不稳定 | exp_reduction_logsumexp_001 | 35% | log-sum-exp trick：先减全局 max | numerical_stability.md |
| exp 溢出保护 | exp_reduction_exp_protection_001 | 25% | exp 前减去局部 max，防止数值溢出 | numerical_stability.md |
| cross-entropy 计算 | exp_reduction_cross_entropy_001 | 30% | 融合 log-softmax + cross-entropy | numerical_stability.md |
| argmax 与 max 联合 | exp_reduction_argmax_001 | 25% | 单遍同时计算 max 值和 argmax 索引 | numerical_stability.md |
| 大 K 并行（K≥4096） | exp_splitk_reduction_001 | 35% | split-K：K 维按核划分 + atomic_add 归约 | parallel_reduction.md |
| 大 K 两阶段方案 | exp_splitk_two_pass_001 | 40% | Phase1 部分和，Phase2 全局规约 | parallel_reduction.md |
| 超大张量多遍规约 | exp_multipass_reduction_001 | 30% | 多遍递进规约（层次化） | parallel_reduction.md |
| 直方图原子 binning | exp_histogram_reduction_001 | 25% | 原子操作 binning，寄存器局部聚合后写回 | parallel_reduction.md |
| 累加和（前缀扫描） | exp_cumsum_001 | 25% | scan primitive 实现 cumsum | parallel_reduction.md |

## 关键技术

### FP32 累加（避免溢出）

fp16 最大值约 65504，在大 K 场景下累加极易溢出；bf16 精度更低（7 位尾数）。始终在 fp32 中累加：

```python
@triton.jit
def sum_kernel(x_ptr, y_ptr, M, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)  # fp32 累加器
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_row + k_offs, mask=mask, other=0.0)
        acc += x.to(tl.float32)          # 提升到 fp32

    result = tl.sum(acc)
    tl.store(y_ptr + row, result.to(tl.float16))
```

### Log-Sum-Exp Trick

直接计算 `log(sum(exp(x)))` 在大值输入时 exp 溢出。通过减去全局 max 保证数值稳定：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - max_x) * exp(max_x)))
= max_x + log(sum(exp(x_i - max_x)))
```

```python
@triton.jit
def logsumexp_kernel(x_ptr, y_ptr, M, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * K

    # Pass 1: 求全局 max
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_row + k_offs, mask=mask, other=float('-inf')).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x))

    # Pass 2: 减去 max 后 exp + sum
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_row + k_offs, mask=mask, other=float('-inf')).to(tl.float32)
        sum_exp += tl.sum(tl.exp(x - max_val) * mask.to(tl.float32))

    y = max_val + tl.log(sum_exp)
    tl.store(y_ptr + row, y.to(tl.float32))
```

### Split-K 并行归约（atomic_add 方案）

对 K ≥ 4096 的归约，将 K 维拆分到多个 AI Core，用 atomic_add 直接归约（exp_splitk_reduction_001，35% 提升）：

```python
@triton.jit
def splitk_atomic_kernel(x_ptr, y_ptr, M, K,
                          BLOCK_K: tl.constexpr,
                          SPLIT: tl.constexpr):
    # grid = (M, SPLIT)
    row = tl.program_id(0)
    split_id = tl.program_id(1)

    k_per_split = tl.cdiv(K, SPLIT)
    k_start = split_id * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        mask = k_offs < k_end
        x = tl.load(x_ptr + row * K + k_offs, mask=mask, other=0.0).to(tl.float32)
        acc += x

    partial_sum = tl.sum(acc)
    # 原子加到输出（自动全局归约）
    tl.atomic_add(y_ptr + row, partial_sum)


def splitk_reduction(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    y = torch.zeros(M, dtype=torch.float32, device=x.device)
    SPLIT = min(32, triton.cdiv(K, 128))  # 根据 K 大小动态确定 SPLIT
    BLOCK_K = 128
    grid = (M, SPLIT)
    splitk_atomic_kernel[grid](x, y, M, K, BLOCK_K=BLOCK_K, SPLIT=SPLIT)
    return y.to(x.dtype)
```

### Split-K 两阶段方案（exp_splitk_two_pass_001，40% 提升）

atomic_add 有竞争开销。两阶段方案：Phase 1 各核写入独立 partial buffer；Phase 2 单独内核做最终归约：

```python
@triton.jit
def splitk_phase1_kernel(x_ptr, partial_ptr, M, K,
                          BLOCK_K: tl.constexpr,
                          SPLIT: tl.constexpr):
    row = tl.program_id(0)
    split_id = tl.program_id(1)
    k_per_split = tl.cdiv(K, SPLIT)
    k_start = split_id * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        mask = k_offs < k_end
        x = tl.load(x_ptr + row * K + k_offs, mask=mask, other=0.0).to(tl.float32)
        acc += x

    # 写入 partial buffer：shape [M, SPLIT]
    tl.store(partial_ptr + row * SPLIT + split_id, tl.sum(acc))


@triton.jit
def splitk_phase2_kernel(partial_ptr, y_ptr, M, SPLIT,
                          BLOCK_SPLIT: tl.constexpr):
    row = tl.program_id(0)
    split_offs = tl.arange(0, BLOCK_SPLIT)
    mask = split_offs < SPLIT
    partials = tl.load(partial_ptr + row * SPLIT + split_offs, mask=mask, other=0.0)
    tl.store(y_ptr + row, tl.sum(partials))
```

### 联合 Argmax + Max（单遍）

单遍同时维护最大值及其索引，避免两遍扫描（exp_reduction_argmax_001，25% 提升）：

```python
@triton.jit
def argmax_kernel(x_ptr, val_ptr, idx_ptr, M, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * K

    running_max = tl.full([1], float('-inf'), dtype=tl.float32)
    running_idx = tl.zeros([1], dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_row + k_offs, mask=mask, other=float('-inf')).to(tl.float32)

        block_max = tl.max(x)
        # 找到 block 内 argmax
        is_max = (x == block_max) & mask
        block_idx = tl.sum(tl.where(is_max, k_offs, 0))

        # 与全局 running max 合并
        update = block_max > running_max
        running_max = tl.where(update, block_max, running_max)
        running_idx = tl.where(update, block_idx, running_idx)

    tl.store(val_ptr + row, running_max.to(tl.float16))
    tl.store(idx_ptr + row, running_idx)
```

### Fused Cross-Entropy（含 log-softmax）

cross-entropy 与 log-softmax 融合：单遍计算 logsumexp，直接输出 loss，无需中间 softmax 张量（exp_reduction_cross_entropy_001，30% 提升）：

```python
@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, loss_ptr,
                          M, V, BLOCK_V: tl.constexpr):
    row = tl.program_id(0)
    logits_row = logits_ptr + row * V

    # Pass 1: max（数值稳定）
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        mask = v_offs < V
        x = tl.load(logits_row + v_offs, mask=mask, other=float('-inf')).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x))

    # Pass 2: sum(exp(x - max))
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        mask = v_offs < V
        x = tl.load(logits_row + v_offs, mask=mask, other=float('-inf')).to(tl.float32)
        sum_exp += tl.sum(tl.exp(x - max_val) * mask.to(tl.float32))

    log_sum_exp = max_val + tl.log(sum_exp)

    # 取 ground truth logit
    label = tl.load(labels_ptr + row)
    gt_logit = tl.load(logits_row + label).to(tl.float32)

    loss = log_sum_exp - gt_logit
    tl.store(loss_ptr + row, loss.to(tl.float32))
```

### Cumulative Sum（前缀扫描）

使用 scan primitive 实现 cumsum（exp_cumsum_001，25% 提升）：

```python
@triton.jit
def cumsum_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # Triton 内置 scan（前缀扫描）
    y = tl.cumsum(x, axis=0)
    tl.store(y_ptr + offsets, y, mask=mask)
```

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| fp16/bf16 累加溢出 | K 较大时 fp16 sum 溢出（最大 65504） | 使用 fp32 累加器（exp_reduction_fp32_001） |
| exp 直接溢出 | 未减 max 直接 exp(x) 溢出 | 使用 log-sum-exp trick（exp_reduction_logsumexp_001） |
| split-K atomic 竞争 | 多核 atomic_add 到同一地址，竞争严重时性能下降 | 改用两阶段方案（exp_splitk_two_pass_001） |
| split-K SPLIT 过大 | SPLIT > AI Core 数时产生资源浪费和同步开销 | SPLIT ≤ num_aicores，通常 SPLIT ≤ 32 |
| argmax 两遍扫描 | 先 max 再 argmax 需要读两遍输入 | 单遍联合 argmax+max（exp_reduction_argmax_001） |
| logsumexp 精度下溢 | max 与 x_i 差距极大时 exp(x_i - max) 下溢为 0 | 保持 fp32 精度，不在 fp16 下计算 log-sum-exp |
| 小 K 使用 split-K | K < 512 时 split-K 的 launch 和同步开销超过收益 | K < 512 时用单核单遍归约 |
| 忽略 histogram 冲突 | 多核同时写同一 bin，atomic 竞争严重 | 寄存器局部 histogram，最后单次写回（exp_histogram_reduction_001） |

## 代码模板

### 通用行归约模板（sum/max/min）

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({'BLOCK_K': bk}) for bk in [128, 256, 512, 1024]],
    key=['K'],
)
@triton.jit
def row_reduction_kernel(
    x_ptr,
    y_ptr,
    M, K,
    BLOCK_K: tl.constexpr,
    REDUCE_OP: tl.constexpr,  # 'sum' | 'max' | 'min'
):
    row = tl.program_id(0)
    x_row = x_ptr + row * K

    if REDUCE_OP == 'sum':
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    elif REDUCE_OP == 'max':
        acc = tl.full([BLOCK_K], float('-inf'), dtype=tl.float32)
    else:
        acc = tl.full([BLOCK_K], float('inf'), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_row + k_offs, mask=mask,
                     other=(0.0 if REDUCE_OP == 'sum' else float('-inf'))).to(tl.float32)
        if REDUCE_OP == 'sum':
            acc += x
        elif REDUCE_OP == 'max':
            acc = tl.maximum(acc, x)
        else:
            acc = tl.minimum(acc, x)

    if REDUCE_OP == 'sum':
        result = tl.sum(acc)
    elif REDUCE_OP == 'max':
        result = tl.max(acc)
    else:
        result = tl.min(acc)

    tl.store(y_ptr + row, result.to(tl.float16))


def row_sum(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    y = torch.empty(M, dtype=x.dtype, device=x.device)
    row_reduction_kernel[(M,)](x, y, M, K, REDUCE_OP='sum')
    return y
```

### Split-K 归约完整示例

```python
def splitk_sum(x: torch.Tensor, split: int = 16) -> torch.Tensor:
    """两阶段 split-K sum，适用于 K >= 4096。"""
    M, K = x.shape
    BLOCK_K = 256

    # Phase 1: 各核计算部分和
    partial = torch.zeros(M, split, dtype=torch.float32, device=x.device)
    splitk_phase1_kernel[(M, split)](x, partial, M, K,
                                      BLOCK_K=BLOCK_K, SPLIT=split)

    # Phase 2: 全局归约
    y = torch.empty(M, dtype=x.dtype, device=x.device)
    BLOCK_SPLIT = triton.next_power_of_2(split)
    splitk_phase2_kernel[(M,)](partial, y, M, split, BLOCK_SPLIT=BLOCK_SPLIT)
    return y
```

## 相关页面

- [normalization.md](op-class-normalization.md) — 归一化算子（LayerNorm、RMSNorm、Softmax）内部大量使用归约原语
- [elementwise.md](op-class-elementwise.md) — softmax = exp（elementwise）+ sum_reduce + div（elementwise）
- [gather_scatter.md](op-class-gather_scatter.md) — MoE 场景中 wgrad 需要 scatter 后的 sum 归约

## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `018-_reduce_split_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/018-_reduce_split_kernel.py`）
- `022-count_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/022-count_kernel.py`）
- `114-sum_and_scale.py` — extracted kernel（source: `raw_experiences/extracted_kernels/114-sum_and_scale.py`）
- `134-bitonic_sortbykey_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/134-bitonic_sortbykey_kernel.py`）
- `136-digit_hist_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/136-digit_hist_kernel.py`）
- `138-duplicate_keys_shuffle_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/138-duplicate_keys_shuffle_kernel.py`）
- `139-multinomial_with_replacement.py` — extracted kernel（source: `raw_experiences/extracted_kernels/139-multinomial_with_replacement.py`）
- `165-rejection_greedy_sample_spec_len_1_triton.py` — extracted kernel（source: `raw_experiences/extracted_kernels/165-rejection_greedy_sample_spec_len_1_triton.py`）
- `167-rejection_greedy_sample_triton.py` — extracted kernel（source: `raw_experiences/extracted_kernels/167-rejection_greedy_sample_triton.py`）
- `168-rejection_random_sample_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/168-rejection_random_sample_kernel.py`）
- `170-sample_recovered_tokens_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/170-sample_recovered_tokens_kernel.py`）
- `171-rejection_random_sample_block_verify_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/171-rejection_random_sample_block_verify_kernel.py`）
- `186-chunk_local_cumsum_scalar_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/186-chunk_local_cumsum_scalar_kernel.py`）
- `222-_top_p_sample_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/222-_top_p_sample_kernel.py`）
- `223-_top_p_filter_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/223-_top_p_filter_kernel.py`）
- `224-_reject_sampler_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/224-_reject_sampler_kernel.py`）
- `225-_join_prob_reject_sampler_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/225-_join_prob_reject_sampler_kernel.py`）

<!-- okf:related:start -->

# 相关

- 讲解接口: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability
- 讲解接口: [sqrt 前用 max(x, eps) 钳位防负数开方 NaN（eps 按精度选型）](../runbooks/algorithm-numerics/exp_stability_002.md) — 同属优化知识，主题相关
- 讲解接口: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../runbooks/algorithm-numerics/fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability, reduction
- 讲解接口: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 numerical_stability
- 讲解接口: [标量消除: min() 运行时分支改 tl.where](../runbooks/compute-pipeline/exp_perf_013.md) — 同属优化知识，主题相关
- 讲解接口: [Memory Access Optimization: 2D block tiling with partial-sum reduction](../runbooks/memory-sram/exp_rmsnorm_grad_dw_001.md) — 同属优化知识，主题相关
- 讲解接口: [pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)](../runbooks/memory-sram/pergroup-blockk-align.md) — 同属优化知识，主题相关
- 相关主题: [Fused Attention / Flash Attention 高性能算子](op-class-attention.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [卷积算子优化（Causal Conv1D）](op-class-convolution.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Elementwise 算子优化](op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [融合算子综合页](op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, numerical_stability, operator_class
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Softmax / LogSoftmax 算子优化](op-class-softmax.md) — 主题相关：共享 910b2, 910b3, numerical_stability, operator_class, reduction
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [SSM / 状态空间模型算子优化](op-class-ssm.md) — 主题相关：共享 910b2, 910b3, operator_class

<!-- okf:related:end -->
