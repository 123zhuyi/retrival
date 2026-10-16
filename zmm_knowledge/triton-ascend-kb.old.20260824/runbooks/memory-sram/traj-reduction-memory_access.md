---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: reduction
dimension: "1.1"
title: Sum-and-Scale 归约内存访问优化
description: 针对单 block 内 sum-and-scale 归约，通过放大 BLOCK_N、向量宽加载与循环展开，匹配 910B 内存突发长度，提升 30%。
tags: [reduction, memory_access, sum_and_scale, vectorization, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/reduction/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Sum-and-Scale 归约内存访问优化

## 概述

`sum_and_scale` 是训练框架中常见的轻量归约：对输入求和后再除以一个标量或权重和。该实现运行在 **单个 program** 内，通过一维循环覆盖整个 `N` 维。虽然这限制了多核扩展性，但在 `N` 为几千到几万的场景下，通过内存访问层面的优化仍可获得显著收益。

核心思路是在单核内把内存吞吐打满：

- 把 `BLOCK_N` 从默认 128 提升到 1024 或 2048，匹配 Ascend 910B 的内存突发长度与缓存线，减少事务数量；
- `tl.load` 使用完整向量宽度，单条指令搬运更多数据；
- 对主循环做隐式/显式展开，降低分支预测与循环控制开销；
- 中间累加器固定为 `tl.float32`，保证统计归约精度。

收益在 `N ≥ 1024` 时约 30%；当 `N > 10000` 时，建议进一步拆分为多 block 并行归约，否则单核成为瓶颈。

## 共同模式

### 1. BLOCK_N 与硬件突发长度对齐

`BLOCK_N` 决定每次循环加载的连续元素个数。Ascend 910B 的内存子系统对长突发（burst）访问更高效；`BLOCK_N=1024/2048` 通常比 128 能减少约 75% 的内存事务和循环控制指令。但过大会增加寄存器占用，需与 kernel 内其他 live variable 权衡。

### 2. 向量宽度最大化的连续加载

`offset = off + tl.arange(0, BLOCK_N)` 生成连续索引，`tl.load(inp_ptr + offset, mask=...)` 让编译器生成合并的向量 load。只要基地址对齐，单次 load 即可达到硬件最大向量宽度，而不是拆成多条窄指令。

### 3. 循环展开降低控制开销

Triton 编译器会对 `for off in range(0, N, BLOCK_N)` 做一定展开，但当迭代次数由运行时 `N` 决定时，展开程度受限。若 `N` 在编译期已知或通过 `tl.constexpr` 传入，可进一步消除循环变量更新与边界判断。

### 4. fp32 累加器 + 延迟除法

无论输入是 fp16/bf16/fp32，`mid_sum` 与 `mid_wgt` 均使用 `tl.float32`。最终 `out_val /= scale_val` 只在单个标量上进行一次，避免低精度大数组累加的数值漂移。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_sum_scale_001` | 30% | 单 block sum-and-scale 通过放大 BLOCK_N、向量加载与循环展开匹配 910B 内存突发，提升 30%。 |

## 代表 Kernel

以下为 `exp_sum_scale_001` 对应的 kernel：

```python
@libentry()
@triton.jit
def sum_and_scale(
    inp_ptr,
    out_ptr,
    N,
    scalebyw: tl.constexpr,
    BLOCK_N: tl.constexpr = 128,
    scale=1.0,
    mean_num=None,
):
    mid_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    if scalebyw:
        mid_wgt = tl.zeros([BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            mask = offset < N
            inp_vals = tl.load(inp_ptr + offset, mask=mask, other=0.0)
            mid_sum += inp_vals
            wgt_vals = tl.load(scale + offset, mask=mask, other=0.0)
            mid_wgt += wgt_vals
        out_val = tl.sum(mid_sum)
        scale_val = tl.sum(mid_wgt)
        tl.store(mean_num, scale_val)
    else:
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            mask = offset < N
            inp_vals = tl.load(inp_ptr + offset, mask=mask, other=0.0)
            mid_sum += inp_vals
        out_val = tl.sum(mid_sum)
        scale_val = scale
    out_val /= scale_val
    tl.store(out_ptr, out_val)
```

## 已知实例

- `exp_sum_scale_001` — Memory Access Optimization（收益 30%）
  - 适用于单 block sum-and-scale 归约，关键参数 `N ≥ 1024`，推荐 `BLOCK_N = 1024/2048`。
  - 对应 kernel: `raw_experiences/extracted_kernels/114-sum_and_scale.py`

## 关联知识

- 开发指南: [Reduction 算子优化](../../reference/op-class-reduction.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, vectorization
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3, reduction
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, reduction
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 reduction
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md) — 主题相关：共享 reduction
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Activation Scheduling 优化聚合](../compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Scatter Weight Gradient 向量化计算优化](../compute-pipeline/traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](../compute-pipeline/traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, reduction
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, vectorization
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, reduction
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
