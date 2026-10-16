---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 输入进入 kernel 前必须 contiguous
description: host 侧 forward 中调用 x.contiguous()，避免非连续张量导致 kernel 内 stride 计算复杂化
tags: [contiguous, stride, memory_access, host_preprocess]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-2 输入 contiguous 化

【优化维度】：内存｜访存效率
【泛化层级】：Triton通用（迁移条件：所有 kernel 输入张量；无失效边界）

<原则：非连续张量在 kernel 内需要额外 stride 计算和分散访存，破坏 GM->UB 连续 coalescing。host 侧 forward 进入 kernel 前统一调用 .contiguous() 保证连续访存。>

<骨架（占位名伪码）：
# host 侧 forward
def forward(self, input_tensor, ...):
    input_tensor = input_tensor.contiguous()   # 进入 kernel 前保证连续
    kernel[grid](input_tensor, ...)
>

**坏实践（反例）**：直接传非连续张量给 kernel
```python
def forward(self, x, ...):
    kernel[grid](x, ...)   # x 可能是非连续（如 transpose 后）
# kernel 内需要 stride 计算，访存分散
```
后果：kernel 内 stride 计算复杂化，破坏连续 coalescing，GM->UB 访存效率下降。✅ 正解即 host 侧 .contiguous()。

**已知实例**：tensor-transform G7（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — 主题相关：共享 stride
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](../compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 主题相关：共享 stride
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 主题相关：共享 contiguous
- 相关主题: [禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）](../portability/no-pytorch-in-forward.md) — 主题相关：共享 host_preprocess
- 相关主题: [多维输入必须在 host 侧扁平化为 2D](host-flatten-to-2d.md) — 主题相关：共享 host_preprocess
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](normalize-dim-to-last.md) — 主题相关：共享 host_preprocess
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 contiguous, memory_access, stride
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 memory_access

<!-- okf:related:end -->
