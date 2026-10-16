---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: 多核负载均衡分配公式（每 core 处理 block 数差不超过 1）
description: 按输出元素总数分配核数，前 remainder 个 core 多处理一个 block，确保负载均衡
tags: [load_balance, block_per_core, remainder, even_split]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-4 多核负载均衡分配公式

【优化维度】：计算｜多核负载
【泛化层级】：Triton通用（迁移条件：total_blocks 不能被 num_cores 整除；失效边界：整除时简单均分即可）

<原则：total_blocks 不能被 num_cores 整除时，简单均分会让部分 core 多干活，尾部 core 空闲。用余数分配：前 remainder 个 core 多处理一个 block，确保每 core 处理 block 数差不超过 1。grid 钳制见 [[grid-cap-to-core-num]]，核数读取见 [[dynamic-vector-core-num]]。>

<骨架（占位名伪码）：
blocks_per_core = total_blocks // num_cores
remainder = total_blocks - blocks_per_core * num_cores
if pid < remainder:
    my_blocks = blocks_per_core + 1
    start_block = pid * (blocks_per_core + 1)
else:
    my_blocks = blocks_per_core
    start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core
>

**坏实践（反例）**：简单均分不处理余数
```python
my_blocks = total_blocks // num_cores   # 丢掉余数，部分 block 没人处理
start_block = pid * my_blocks
```
后果：余数 block 无人处理，输出缺数据；或尾部 core 空闲等待。✅ 正解即余数分配公式。

**已知实例**：tensor-transform G6（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](../memory-sram/blockm-256-large-m.md) — 同属优化知识，主题相关
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](../memory-sram/pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../memory-sram/traj-embedding-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](../memory-sram/ub-aware-block-adaptive.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 load_balance
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](scalar-load-channel-index.md) — 同属优化知识，主题相关
- 相关主题: [swizzle2d 块重排提升 L2 复用](swizzle2d-block-reorder.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 大 Shape 算子的交错循环负载均衡](traj-general_ascend-load_balance.md) — 主题相关：共享 load_balance
- 相关主题: [Triton Block Size 通用调优原则](traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 相关主题: [Work-Stealing 动态负载均衡（不均匀任务分发）](work-stealing-dynamic-load-balance.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
