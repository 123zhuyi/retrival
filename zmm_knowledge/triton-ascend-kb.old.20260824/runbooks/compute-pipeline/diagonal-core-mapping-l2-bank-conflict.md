---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: 大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict
description: 默认行列 block 映射会让相邻 program 访问同一 L2 bank，大矩阵 matmul 上用对角 core mapping 把访问分散到不同 bank，降低 aiv_mte2_time。
tags: [bank_conflict, l2_cache, diagonal_mapping, core_mapping, matmul, aiv_mte2]
confidence: verified
status: verified
created_at: '2026-08-10T00:00:00Z'
updated_at: '2026-08-10T00:00:00Z'
---

## OPT-19 对角 core mapping 消除 L2 bank conflict

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：2D block 划分的大矩阵 matmul，相邻 program 共享输入块导致 L2 bank conflict；失效边界：1D 划分、矩阵过小或访问模式本身已分散）

<原则：Ascend L2 cache 分多 bank。默认按列/行映射时，相邻 block_id 访问同一 B 列或相邻 A 行，多个 core 同时命中同一 bank，访问被串行化，有效带宽下降。对角映射让相邻 program 落到不同 (M, N) 位置，分散 L2 bank 访问，降低 `aiv_mte2_time`。>

**为什么相邻 PIDs 会冲突（示意图）**：

```
默认 Column 映射（冲突）：
Core 0: A[row=0..127, :]   × B[:, col=0..127]   ← 与 Core 1 读同一 B 列
Core 1: A[row=128..255, :] × B[:, col=0..127]   ← 与 Core 0 读同一 B 列
Core 2: A[row=256..383, :] × B[:, col=0..127]   ← 与 Core 0/1 读同一 B 列
结果：Core 0/1/2 同时读同一 L2 bank → 串行化

对角映射（无冲突）：
Core 0: A[row=0..127, :]   × B[:, col=0..127]    ← 唯一 B 列
Core 1: A[row=128..255, :] × B[:, col=128..255]  ← 唯一 B 列
Core 2: A[row=256..383, :] × B[:, col=256..383]  ← 唯一 B 列
结果：每个 core 读不同 L2 bank → 无串行化
```

<骨架（占位名伪码）：
import triton_ascend.language as dl

pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)

# 对角重映射：相邻 pid -> 不同 (M, N) bank
pid_m, pid_n = dl.grouped_launch_diagonal(pid, num_pid_m, num_pid_n, GROUP_SIZE)

# 后续按 pid_m / pid_n 计算 base 指针
rm = pid_m * BLOCK_M
rn = pid_n * BLOCK_N
>

**诊断阈值**：

```
Theoretical_mte2_time = Data_volume / HBM_bandwidth
                      = (M*K + K*N + M*N) * dtype_bytes / 1.8e12
```

| 实测 / 理论 | 解读 |
|---|---|
| ≈ 1x | 访存高效，无明显 bank conflict |
| 2–5x | 存在 bank conflict 或不对齐 |
| > 5x | 严重 bank conflict 或完全未对齐 |

**触发条件**：`M ≥ 512` 且 `N ≥ 512` 的大矩阵 matmul 优先启用；若同时伴随 512B 行宽未对齐，问题会叠加（先修对齐再修 mapping）。

**GROUP_SIZE 选择**：
- 默认 `GROUP_SIZE = 8`：同组内相邻 PID 共享 K 维切片，提升 L2 复用。
- `K` 很大时尝试 `16` 或 `32`：进一步增加 K 维复用，但会降低组内 spatial diversity，需 autotune 验证。

**诊断清单**：
1. 跑 profiling 拿到 `aiv_mte2_time`。
2. 用上面公式算 theoretical time。
3. 比值 > 2x → 怀疑 bank conflict。
4. 确认 `M ≥ 512 AND N ≥ 512` → 应用对角映射。
5. 确认行宽 512B 对齐（见 [Cube-Vector 行宽 512B 对齐](../memory-sram/row-width-512b-align.md)）。
6. 重跑 profiling 确认 `aiv_mte2_time` 回落。

**坏实践（反例）**：
```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n   # 线性行列映射
```
后果：相邻 program 访问同一 B 列，L2 bank conflict，有效带宽下降。✅ 正解即 `dl.grouped_launch_diagonal` 对角映射。

> **注意**：Ascend 上 `tl.swizzle2d` 行为未定义，应使用 `triton_ascend.language.grouped_launch_diagonal` 作为 L2 bank conflict 的规避手段。与 [swizzle2d 块重排](swizzle2d-block-reorder.md) 目标相同（提升 L2 复用/降低冲突），但实现原语不同。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_matmul_diag_tiling_001` — Scheduling Optimization （收益 25%）
  - 本优化专为昇腾 910B 大规模矩阵场景设计，核心目标是缓解内存带宽压力与 L2 Cache 容量 瓶颈。对角线分核在 M、N 方向均超过 BLOCK_TRESHHOLD 个基本块时生效，右矩阵大小超过 L2 Cache 容量时优化效果尤为显著。输出结果以 bfloat16…

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 实践案例: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 matmul
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Matmul 硬件适配优化聚合](../portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 core_mapping, diagonal_mapping, matmul
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 matmul
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](traj-matmul-split.md) — 主题相关：共享 matmul

<!-- okf:related:end -->
