---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.4"
title: 复合块多 kernel pipeline 分解模式（~30-op conv+transformer 块的已验证骨架）
description: conv+unfold+N×(Attn+FFN)+fold+conv fusion 复合块单 kernel 展平不可行；已验证骨架 = universal GEMM（8 个 GEMM 统一，epilogue 可选融合）+ LN kernel + dual-path flash-attn + host 布局视图。
tags: [composite_block, kernel_decomposition, universal_gemm, epilogue_fusion, multi_kernel_pipeline, mobilevit]
confidence: draft
status: draft
created_at: "2026-08-26T00:00:00Z"
updated_at: "2026-08-26T00:00:00Z"
---

## 复合块多 kernel pipeline 分解（来源：53_MobileViTAttention 1.5979x，49/49）

【泛化层级】：条件性（迁移条件：conv+transformer 类复合块；失效边界见末节）

**骨架（已验证架构决策）**：
1. **universal GEMM kernel**：conv1/conv2/qkv/out/ff1/ff2/conv3/conv4 共 8 个 GEMM 统一为一个 kernel，可选 bias/residual/silu-on-A prologue 与 silu epilogue 融合（SiLU 移入 ff1 epilogue 可消 num_n 次重复计算）；
2. **行并行 LN kernel**：fp32 统计 + rsqrt；BM 受 UB 限（BM×512×4B ≤ 32KB → BM=16；BM=32 会 MLIRCompilationError）；
3. **dual-path flash-attn**：按 n（token 数）分派——多块路径（n 大）与 onekv 路径（n≤np2 窗口），窗口由 UB 定量式决定（见 device-side-gather 卡）；
4. **host 布局视图**：reshape/permute/unfold/fold 留 host 零拷贝，不进 kernel；
5. **grid = min(natural_blocks, VEC_CORE_NUM)**，num_cores 入参 = grid_size，禁止 grid_size > VEC_CORE_NUM。

**UB 预算**：GEMM BM=256/BK=128 fp32 顶满 192KB（局部最优，BK 降 64 会因 K-loop 翻倍抵消双缓冲，-1.5%）。

**已证伪方向（勿重试）**：qkv 直写窗口（n∈(48,64]，负收益）；Triton im2col（嵌套 runtime 循环 + 2D masked load 触发 UB 结构性放大 ~24x，0.09x 灾难）；SCALE 预乘进 q（+8.4% 回退，破坏 dot 流水）；hf32 dot（两次零增益，疑似静默 ieee fallback）。

**失效边界**：UB 超 192KB 的层间真融合（如 FFN 中间 [64,1024] fp32 = 256KB，不可行）；target_speedup ≥ 2.0 在 fp32 精度契约 + 192KB UB 下对本类算子不可达（三方互证），设定目标时参考 ~1.4-1.6x 为现实上限。

<!-- okf:related:start -->

# 相关

- 相关主题: [fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）](../algorithm-numerics/fp32-precision-contract.md) — 主题相关：共享 composite_block, mobilevit

<!-- okf:related:end -->
