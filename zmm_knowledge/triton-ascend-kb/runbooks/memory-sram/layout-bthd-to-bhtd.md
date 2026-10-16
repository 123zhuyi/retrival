---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存
description: 序列维分块迭代类算子（GDN/KDA/Mamba 等 chunkwise 递推）在 (B,T,H,D) layout 下沿 T 分块读取时步长为 H*D、跨 H 维离散访存；在 GM 侧预转置为 (B,H,T,D) 使每次搬运连续，单算子典型提升 38%~79%。
tags: [layout_transform, discrete_memory_access, bthd_bhtd, cache_hit, gdn, linear_attention, chunked_recurrence]
confidence: draft
status: draft
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-08-27T00:00:00Z"
---

## layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存（来源：GDN 三篇优化实践互证）

【优化维度】：内存｜离散访存消除
【泛化层级】：条件性（迁移条件：算子沿序列维 T 分块迭代、且输入 layout 为 (B,T,H,D)/(B,T,H)；失效边界：算子本身按 H 维分核且每核内 T 维访问本就连续，或 transpose 本身的开销超过收益——短序列小 H 场景需实测）

<原则：输入 q/k/v 为 (B,T,H,D) 时，沿 T 维按 BT 分块读 BT*D 数据的步长是 H*D，每次搬运跨越 H 维，Cache 命中率低、MTE2 效率差。预转置为 (B,H,T,D) 后，同一 (b,h) 内 T 维完全连续，分块读取变为纯顺序搬运。g/beta 等 (B,T,H) 张量同理转 (B,H,T)。>

**正例（Triton-Ascend 可直接落地）**：

```python
# 调用侧一次性转置（GM 侧），kernel 内按 (b, h, t_block) 连续寻址
q = q.transpose(1, 2).contiguous()   # (B,T,H,D) -> (B,H,T,D)
k = k.transpose(1, 2).contiguous()
v = v.transpose(1, 2).contiguous()

@triton.jit
def chunk_fwd(Q, K, V, ..., H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    # 基址 = ((b * H + h) * T + t0) * D，块内 stride 为 1
    offs = ((b * H + h) * T + t0) * D + tl.arange(0, BT)[:, None] * D + tl.arange(0, D)[None, :]
    q_tile = tl.load(Q + offs)   # 连续访存
```

**坏实践（反例）**：

```python
# (B,T,H,D) 下直接沿 T 分块：行步长 H*D，跨 H 维跳跃
offs = ((b * T + t0) * H + h) * D + tl.arange(0, BT)[:, None] * (H * D) + ...
```

后果：MTE2 搬运离散、Cache 命中差。实测 GDN fwd 算子 (2,8192,8,128) 场景耗时 4.56ms→0.95ms（+79%），三篇独立实践平均 +38%~79%。✅ 正解即 GM 侧预转置。

**注意事项**：

- 转置本身是一次额外 elementwise pass，序列短时可能入不敷出；整网场景下若前后算子可协商 layout，转置开销可摊薄到零。
- chunkSize 从 64 调大到 128 与本优化正交且可叠加（更大连续块 + 更少递推次数），GDN 实测 chunkSize=128 稳定优于 64（Fwd_h 平均 +46.75%）。
- 收益与平台带宽正相关：950DT（高带宽显存）上绝对收益最大。

**收益数据（带实现语境）**：

| 来源 | 实现 | 场景 | 收益 |
|---|---|---|---|
| FLA 移植实践(2025-12) | Triton/A2/A3 | GDN 常用场景 | 单算子平均 +38% |
| Triton 体系化(2026-03) | Triton/A2/A3 | chunk_gated_delta_rule_fwd (2,8192,8,128) | +74.6% |
| AscendC 重写(2026-08) | AscendC/910B1+950 | Fwd_h/Fwd_o 基线对齐前置手段 | 已作为默认 layout |

**Triton 可达性**：✅ 完全可达（纯调用侧 + 寻址方式改动，无 DSL 壁垒）。

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — chunkwise 递推类 layout 预转置直接适用 FwdH
- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 主题相关：共享 chunked_recurrence, gdn, linear_attention
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — chunkwise 递推 layout 预转置适用 Gdn2Fwd
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 linear_attention
- 开发指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 linear_attention
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 主题相关：共享 chunked_recurrence, gdn
- 相关主题: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — 主题相关：共享 gdn
- 相关主题: [Scheduling Optimization: bdt layout rationale](../compute-pipeline/exp_causal_conv1d_update_007.md) — 同属优化知识，主题相关
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../compute-pipeline/load-balance-grid-sizing.md) — 主题相关：共享 gdn
- 相关主题: [swizzle2d 块重排提升 L2 复用](../compute-pipeline/swizzle2d-block-reorder.md) — 主题相关：共享 cache_hit
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../meta-skills/pipeline-diagram-driven-diagnosis.md) — 主题相关：共享 gdn
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](../portability/exp_rope_matmul_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](ub-fill-and-compute-hoist.md) — 主题相关：共享 gdn

<!-- okf:related:end -->
