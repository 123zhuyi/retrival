---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: 负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid
description: GPU 三维 grid 直迁昇腾时，若总程序数超过物理核数（A3 为 24 Cube/48 Vector）会重复发起核调用浪费资源；grid 设为物理核数+核内循环处理任务，或任务数不足时设为实际并行数/其约数，GDN 三个算子实测提升 90%~99.6%。
tags: [grid_sizing, load_balance, physical_cores, persistent_kernel, gpu_to_npu, gdn]
confidence: draft
status: draft
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-08-27T00:00:00Z"
---

## 负载均衡分核（来源：Triton GDN 体系化实践，GPU→NPU 迁移第一课）

【优化维度】：迁移｜任务分核
【泛化层级】：通用（迁移条件：任何从 GPU Triton 迁移的多维 grid kernel；失效边界：grid 本就 ≤ 物理核数且负载均衡的 kernel 无需改动）

<原则：GPU SIMT 习惯 grid=(x,y,z) 三维海量线程块；昇腾核数少（A3：24 Cube + 48 Vector），总程序数 > 物理核数时会多波次重复发起核调用，且 GPU 原生的细粒度分块在 NPU 上每核计算量过少、启动开销占比高。惯例：grid 直接设为物理核数（persistent 风格），核内 `for task in range(pid, n_tasks, num_progs)` 循环领取任务；当任务数 < 物理核数或不能整除时，grid 设为实际可并行任务数或能整除任务数的合理值，避免尾核空转/过载。>

**正例**：

```python
NUM_CORES = 48  # A3 Vector 核数，按 kernel 主力单元选取

@triton.jit
def recompute_w_u(..., n_tasks, ...):
    pid = tl.program_id(0)
    for t in range(pid, n_tasks, tl.num_programs(0)):
        ...  # 核内循环处理多个任务

recompute_w_u[(NUM_CORES,)](..., n_tasks, ...)
```

**坏实践（反例）**：

```python
kernel[(B, H, T // BT)](...)   # GPU 三维 grid 直迁：B*H*n_chunks 程序数远超核数
```

后果：多波次重复发射、每核负载过轻。实测（Triton/A2/A3，相对 FLA 开箱）：recompute_w_u_fwd +90%~95.7%，chunk_fwd_o +98.9%~99.2%，chunk_scaled_dot_kkt_fwd +98.5%~99.6%（注：收益含开箱配置极差的基线效应，代表"迁移时最先做的一刀"）。

**注意事项**：

- 本卡与"大 shape 连续分块访存"（memory-sram）同源：核内循环领取**连续段**任务，兼顾负载均衡与访存局部性。
- 分核数按 kernel 的瓶颈单元选择：Cube 密集型对齐 Cube 核数，Vector 密集型对齐 Vector 核数，CV 融合型需实测。
- chunk 递推类算子（GDN Fwd_h 型）因块间依赖并行度天然受限，B*H 小时即使用满核也可能吃不饱——此时优先大 chunkSize + Head 交错（见 compute-pipeline 的 Pingpong 卡），而非继续加分核。

**Triton 可达性**：✅ 完全可达。

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — GDN 三算子分核实测出处（fwd_o/kkt +99%）
- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 主题相关：共享 gdn
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — 共享 gdn：核数对齐分核适用双骨架
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](../memory-sram/exp_rmsnorm_bwd_fused_001.md) — 同属优化知识，主题相关
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 gdn
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 主题相关：共享 gdn
- 相关主题: [GPU-source 迁移到 Ascend NPU 的 memory/index 改造检查清单](../meta-skills/gpu-to-ascend-migration-checklist.md) — 同属优化知识，主题相关
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../meta-skills/pipeline-diagram-driven-diagnosis.md) — 主题相关：共享 gdn
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](chunk-recurrence-pingpong-dispatch.md) — 主题相关：共享 gdn
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 load_balance
- 相关主题: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](cv-preload-and-bandwidth-arbitration.md) — 主题相关：共享 gdn
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 主题相关：共享 load_balance
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 persistent_kernel
- 相关主题: [Work-Stealing 动态负载均衡（不均匀任务分发）](work-stealing-dynamic-load-balance.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
