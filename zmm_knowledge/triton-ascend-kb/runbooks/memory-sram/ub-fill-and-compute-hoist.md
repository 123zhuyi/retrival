---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧
description: 昇腾 UB 约 192KB，分块调大（如 BV/BK 64→128）可显著提升效率；当 UB 仅微超零点几 KB 时，不要退回小块，而把下三角 mask、g_exp 广播等计算外提到调用侧/GM 融合，省掉 kernel 内中间缓冲，GDN dqkwg 32K 场景 18ms→10ms。
tags: [ub_capacity, tiling, compute_hoist, mask_fusion, g_exp, chunk_size_128, gdn]
confidence: draft
status: draft
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-08-27T00:00:00Z"
---

## UB 打满与外提策略（来源：GDN FLA 移植 + Triton 体系化实践互证）

【优化维度】：内存｜UB 容量利用
【泛化层级】：条件性（迁移条件：UB 192KB 平台上分块受容量限制、且 kernel 内存在可因式分解/可预计算的中间张量（mask、广播 exp 等）；失效边界：UB 超出量大（外提省不出空间），或外提计算无法从 kernel 内循环中剥离——需逐 case 核算 UB 账目）

<原则：GPU 惯用分块 64 是为 warp=32 折中，昇腾 UB 192KB 下合适 Tile 为 16~64KB——BV/BK 从 64 调 128（FP32 下 [64,128]→[128,128]，32KB→64KB）可一次搬运更大块、减少循环与搬运次数。若调大后 UB 微超（如 dqkwg BK=128 需 192.4KB，仅超 0.4KB），不退回小块，而是把 kernel 内的 mask_A（下三角 (BT,BT)）与 g_exp 广播 exp(b_g[:,None]-b_g[None,:]) 外提到调用侧在 GM 融合预计算，kernel 内不再开辟 mask 缓冲。>

**正例（Triton-Ascend）**：

```python
# 调用侧：预计算融合 mask 与 g_exp（GM 侧一次 elementwise pass）
g_diff_exp = torch.exp(g[:, None] - g[None, :]) * tril_mask   # (BT, BT)

@triton.jit
def chunk_bwd_dqkwg(..., GEXP, ...):
    gexp = tl.load(GEXP + offs)   # 直接加载，不再 kernel 内 exp + where + 开缓冲
    ...
```

**坏实践（反例）**：

```python
# BK=128 报 UB overflow 后直接退回 BK=64
```

后果：小块计算效率大打折扣。dqkwg 实测 32K 场景外提后 18ms→10ms；FLA 移植实践常用场景平均 +51%（含外提引入的额外 DIV/EXP/MUL 开销，占比很小）。✅ 正解即外提可预计算项，保住大块。

**配套手段——mask 计算本身也要省**：

- 下三角 mask 的 `tl.where(cond, a, b)` / Select 改为**乘法**（`a * tril_mask`），Vector 上乘法远快于 Select；
- 前序算子已保证输入为下三角时（如 solve_tril 的 A），删除 kernel 内冗余的 `tl.where(m_A, b_A, 0)` 下三角化，符号在 load 阶段附加，实测省约 2ms。

**收益数据（带实现语境）**：

| 来源 | 实现 | 场景 | 收益 |
|---|---|---|---|
| FLA 移植(2025-12) | Triton/A2/A3 | chunk_bwd_dqkwg 常用场景 | 平均 +51% |
| Triton 体系化(2026-03) | Triton/A2/A3 | dqkwg 32K；solve_tril 冗余删除 | 18ms→10ms；省 2ms |
| AscendC(2026-08) | AscendC | Select→乘法 | 基线对齐默认手段 |

**Triton 可达性**：✅ 完全可达（外提、乘法 mask、冗余删除均为 Triton 层改动）。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — GDN dqkwg UB 打满实践，chunk_size_128 同参数域
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 主题相关：共享 gdn
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — 共享 gdn：UB 预算驱动分块（[128,64] 最优即 UB 感知选择）
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 主题相关：共享 gdn
- 相关主题: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — 主题相关：共享 gdn
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 同属优化知识，主题相关
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../compute-pipeline/load-balance-grid-sizing.md) — 主题相关：共享 gdn
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — Part3 UB 超限映射的预算策略卡
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 同属优化知识，主题相关
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../meta-skills/pipeline-diagram-driven-diagnosis.md) — 主题相关：共享 gdn
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — 同属优化知识，主题相关
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](layout-bthd-to-bhtd.md) — 主题相关：共享 gdn

<!-- okf:related:end -->
