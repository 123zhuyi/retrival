---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_perf
dimension: "6.3"
title: 避免 Stride 访问：优先连续化后一维遍历
description: 针对 Ascend 910B 的 stride 内存访问陷阱，说明为何 stride 访问会导致 40% 性能损耗，以及如何通过 contiguous + 一维遍历规避。
tags: [general_perf, perf_pitfall, stride_access, contiguous, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_perf/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# 避免 Stride 访问：优先连续化后一维遍历

## 概述

Triton kernel 中如果直接用 `offsets_m * stride_m + offsets_n * stride_n` 处理二维 stride 张量，虽然语义正确，但在 Ascend 910B 上会带来显著性能惩罚：

- 每次 load/store 都需要计算多维偏移，标量地址运算累积开销随尺寸线性增长；
- stride 访问破坏内存连续性，缓存命中率比连续访问低约 50%；
- 2D 张量（如 1024×1024）的 stride 访问耗时可达连续一维访问的 1.4 倍以上。

该经验的核心建议：

- 进入 kernel 前先把输入 `contiguous()`；
- 用一维 `tl.arange` 按 `numel()` 遍历；
- 只在真正无法避免的非规则布局场景下才使用 stride 访问。

## 共同模式

### 1. Stride 访问的额外开销

对于非连续张量，`data_offsets = offsets_m * stride_m + offsets_n * stride_n` 中的乘加运算需要在每个 program 内逐元素执行。Ascend 的向量单元对规则的一维连续地址更友好，复杂的 stride 地址模式会引入更多标量指令和地址生成延迟。

### 2. 缓存命中率下降

即使计算逻辑相同，stride 访问的内存分布是跳跃的，cache line 利用率低。连续访问可以把相邻数据一次性读入缓存，而 stride 访问可能每条 cache line 只用到少量元素。

### 3. 连续化 + 一维遍历是通用替代

```python
if not input_tensor.is_contiguous():
    input_tensor = input_tensor.contiguous()
output = torch.empty_like(input_tensor)
n_elements = input_tensor.numel()
elementwise_opt_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE=128)
```

在 kernel 内部用 `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` 访问，编译器可以生成最优的 burst load。

### 4. 何时必须保留 stride

某些算子确实需要按原 stride 处理（如 transpose 原地操作、非规则 view 的结果写回）。这种情况下应：

- 尽量使用 `tl.make_block_ptr` 让编译器知道 block 结构；
- 保证 block 边界与 stride 对齐；
- 通过 profiling 确认这是真正的瓶颈，而不是过早优化。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_perf_007` | -40% | Stride 访问相比连续一维访问的性能损耗；在 Ascend 910B 上 2D 张量 stride 访问耗时是连续访问的 1.4 倍以上。 |

## 代表 Kernel

`exp_perf_007` 没有独立的 extracted kernel 文件，以下为 YAML 中的正/反例代码：

```python
# 反例：不推荐的 stride 访问（累积开销大）
@triton.jit
def stride_access_kernel(
    input_ptr, output_ptr, M, N, stride_m, stride_n,
    BLOCK_M: tl.constexpr = 128, BLOCK_N: tl.constexpr = 128,
):
    pid = triton.program_id(0)
    rm = pid // (N // BLOCK_N)
    rn = pid % (N // BLOCK_N)
    offsets_m = rm * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offsets_n = rn * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    data_offsets = offsets_m * stride_m + offsets_n * stride_n
    mask = (offsets_m < M) & (offsets_n < N)
    data = tl.load(input_ptr + data_offsets, mask=mask)
    result = data * 0.5
    tl.store(output_ptr + data_offsets, result, mask=mask)


# 正例：推荐的连续化 + 一维访问（替代 stride）
def replace_stride_access(input_tensor):
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, 128),)
    elementwise_opt_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE=128)
    return output
```

## 已知实例

- `exp_perf_007` — 性能避坑（收益 -40%，表示性能损耗）
  - 适用于 M、N ≥ 128 的 2D 张量；stride 访问在 910b2/910b3 上硬件惩罚显著，优先连续化后一维遍历。

## 关联知识

- 开发指南: [General_Perf 算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [扩展 Autotune 配置空间以覆盖多 Shape 最优调度](../autotuning/traj-general_perf-tuning_config.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](../compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [复杂算子拆分：降低编译与调度开销](../compute-pipeline/traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Element-wise 算子避免冗余 Reshape](../compute-pipeline/traj-general_perf-redundancy_elim.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化](../compute-pipeline/traj-general_perf-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 contiguous
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, contiguous, general_perf
- 相关主题: [输出张量使用 empty_like 替代 zeros_like](../memory-sram/traj-general_perf-memory_init.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Triton-Ascend kernel 调试排查清单](debugging-checklist.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
