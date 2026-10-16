---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_perf
dimension: "1.2"
title: 输出张量使用 empty_like 替代 zeros_like
description: 针对 Triton 算子输出张量创建，优先使用 torch.empty_like() 避免 zero-fill 初始化开销，并保证连续内存布局，可节省约 8% 时间。
tags: [general_perf, memory_init, empty_like, output_tensor, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_perf/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# 输出张量使用 empty_like 替代 zeros_like

## 概述

Triton kernel 通常会自己把输出张量的每个元素写满，因此 host 侧不需要先把它初始化为 0。使用 `torch.empty_like()` 创建输出可以跳过 zero-fill 初始化，并默认获得与输入相同的连续内存布局，直接适配一维向量访问。

核心建议：

- 用 `torch.empty_like(input_tensor)` 创建输出；
- 避免 `torch.zeros_like()` / `torch.ones_like()`，它们会额外触发一次全量内存填充；
- 如果输入不连续，先 `contiguous()` 再计算，不要依赖 zeros_like 的“顺便”连续化。

收益约 8%，在输出张量小于 1e5 元素时初始化开销占比更高。

## 共同模式

### 1. 输出由 kernel 全覆盖时无需预初始化

只要 kernel 内部对每个输出位置都有 `tl.store`，输出张量的初始值就不影响结果。此时 `empty_like` 与 `zeros_like` 语义等价，但前者没有初始化开销。

### 2. empty_like 默认保持连续布局

`torch.empty_like(input)` 会创建与输入同形状、同 dtype、同 strides 的张量。若输入是连续的，输出也是连续的，kernel 内可以直接用一维 `tl.arange` 访问，无需额外 reshape 或 contiguous。

### 3. 避免隐式初始化依赖

有些开发者用 `zeros_like` 是因为担心某些边界位置未被 kernel 写入。更好的做法是：在 kernel 内用 mask 保证所有有效位置都被写入；对真正需要默认值的场景（如 padding），再显式初始化所需区域。

### 4. 小张量场景收益更显著

对于小 tensor，kernel 执行时间本身很短，host 侧 `zeros_like` 的同步初始化可能占到总时间的 5%–10%。`empty_like` 可以消除这部分固定开销。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_perf_008` | 8% | Triton 算子输出张量用 `torch.empty_like()` 替代 `torch.zeros_like()`，避免 zero-fill 初始化，节省约 8% 时间。 |

## 代表 Kernel

`exp_perf_008` 没有独立的 extracted kernel 文件，以下为 YAML 中的 host 调用示例与对应 kernel：

```python
@triton.jit
def elementwise_opt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr = 512):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    output = data * 2 + 1
    tl.store(output_ptr + offsets, output, mask=mask)


class OutputTensorCreateModel(torch.nn.Module):
    def forward(self, input_tensor):
        # 优化：使用 torch.empty_like 创建输出（默认连续，无初始化开销）
        output_tensor = torch.empty_like(input_tensor)

        # 反例：不推荐 - torch.zeros_like 有初始化开销
        # output_tensor = torch.zeros_like(input_tensor)

        input_contig = input_tensor.contiguous() if not input_tensor.is_contiguous() else input_tensor
        n_elements = input_contig.numel()
        grid = (triton.cdiv(n_elements, 512),)
        elementwise_opt_kernel[grid](input_contig, output_tensor, n_elements, BLOCK_SIZE=512)
        return output_tensor
```

## 已知实例

- `exp_perf_008` — 内存初始化优化（收益 8%）
  - 适用于所有 Triton 算子的输出张量创建场景，小张量（<1e5 元素）收益更明显。

## 关联知识

- 开发指南: [General_Perf 算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [扩展 Autotune 配置空间以覆盖多 Shape 最优调度](../autotuning/traj-general_perf-tuning_config.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [复杂算子拆分：降低编译与调度开销](../compute-pipeline/traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Element-wise 算子避免冗余 Reshape](../compute-pipeline/traj-general_perf-redundancy_elim.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化](../compute-pipeline/traj-general_perf-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Ascend 输出张量创建优化：empty/empty_like 替代 zeros/ones](traj-general_ascend-memory_init.md) — 主题相关：共享 910b2, 910b3, memory_init, output_tensor
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, general_perf

<!-- okf:related:end -->
