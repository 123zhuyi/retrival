---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: Work-Stealing 动态负载均衡（不均匀任务分发）
description: 对 MoE Expert 等任务量不均匀场景，用原子计数器动态认领任务，避免静态 grid 导致部分核空闲、部分核过载。
tags: [work_stealing, dynamic_load_balance, atomic_add, moe, expert, uneven_workload]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-25 Work-Stealing 动态负载均衡（不均匀任务分发）

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：任务量不均匀、无法静态均分；失效边界：任务量均匀、或原子操作开销占比过高）

<原则：当不同 program 的处理量差异大时（如 MoE 各 Expert 接收 token 数不同），静态 grid 会导致部分核早早完成空闲、部分核过载。用 `tl.atomic_add` 实现全局任务计数器，每个 program 动态认领下一个任务，可实现近似负载均衡。>

**正例**：

```python
@triton.jit
def work_stealing_kernel(work_queue_ptr, num_tasks, ...):
    while True:
        task_id = tl.atomic_add(work_queue_ptr, 1)  # 原子认领任务
        if task_id >= num_tasks:
            break
        process_task(task_id)
```

**适用场景**：

| 场景 | 说明 |
|---|---|
| MoE Expert 处理 | 各 Expert token 数差异大 |
| 变长序列 batch | 不同序列长度导致计算量不均 |
| 动态路由 | token 到 expert 的映射运行时确定 |
| 任务粒度适中 | 单个任务计算量足够摊还原子操作开销 |

**与静态 grid-stride 的对比**：

| 维度 | 静态 grid-stride | Work-stealing |
|---|---|---|
| 任务均匀 | ✅ 简单高效 | 原子 overhead |
| 任务不均匀 | ❌ 部分核空闲 | ✅ 动态均衡 |
| 实现复杂度 | 低 | 稍高（需原子计数器） |
| 可扩展性 | 受 grid 限制 | 天然支持任意任务数 |

**注意事项**：

- `tl.atomic_add` 本身有 overhead，任务粒度过小会抵消收益。
- 任务数应远大于核数，确保原子竞争不成为瓶颈。
- 需确保 `process_task` 内部无额外同步依赖。

**坏实践（反例）**：
```python
# 静态分配：每个核处理固定数量 expert
expert_id = pid % num_experts
process_expert(expert_id)
```
后果：热门 Expert 核过载，冷门 Expert 核空闲。✅ 正解即动态 work-stealing。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_moe_binned_scatter_001` — Memory Access Optimization （收益 30%）
  - This optimization is specifically designed for Mixture of Experts (MoE) routing layers where tokens must be scattered to specific expert…
  - 对应 kernel: `raw_experiences/extracted_kernels/002-_binned_copy_gather_npu.py`
- `exp_scatter_moe_padded_001` — Memory Access Optimization （收益 30%）
  - This kernel requires that the input bin_ids are sorted by expert ID. The performance gain is most significant when the number of experts is…
  - 对应 kernel: `raw_experiences/extracted_kernels/033-_padded_scatter_kernel.py`

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 atomic_add
- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 主题相关：共享 moe
- 实践案例: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 主题相关：共享 moe
- 实践案例: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](../../ops/moe-init-routing-v2.md) — 主题相关：共享 expert, moe
- 实践案例: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](../../ops/moe-init-routing-v3.md) — 主题相关：共享 moe
- 实践案例: [MoeInitRouting (MoE routing) on Ascend NPU](../../ops/moe-init-routing.md) — 主题相关：共享 expert, moe
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 主题相关：共享 moe
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 主题相关：共享 atomic_add
- 相关主题: [原子操作粒度--连续段 E>=64 用向量 atomic_add，禁止标量循环 atomic](../algorithm-numerics/atomic-granularity.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](scalar-load-channel-index.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
