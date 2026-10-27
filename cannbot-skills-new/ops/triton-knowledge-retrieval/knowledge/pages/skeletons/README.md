---
title: 代码骨架索引
category: skeletons
tags: [skeleton, template, code]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# 代码骨架库

> 完整可运行的 Triton-Ascend kernel 骨架。LLM 生成新算子时以最近的骨架为基础修改，减少从零搭建的错误。

| 骨架文件 | 适用场景 | 关键特性 | 参考经验 |
|---------|---------|---------|---------|
| [gemm_basic.py](gemm_basic.py) | 标准 GEMM，M,N,K≥256 | 对角 tiling + autotune，含融合激活/bias 注释 | exp_matmul_001 (40%) |
| [flash_attention_causal.py](flash_attention_causal.py) | 因果 Flash Attention，支持 GQA | Online softmax，BLOCK_M/N autotune | exp_fused_attention_002 (50%) |
| [rmsnorm_fused.py](rmsnorm_fused.py) | RMSNorm，支持 Skip Add 融合 | Persistent kernel，大 hidden 列分块 two-pass | exp_rmsnorm_001(086) (35%) |
| [moe_dispatch.py](moe_dispatch.py) | MoE Scatter + Expert GEMM + Gather | Binned dispatch，三阶段结构 | exp_gs_bin_003 (35%) |

## 使用方法

1. 根据算子类型选骨架（参考 [workflows/shape_router](../workflows/shape_router.md)）
2. 修改维度参数（M/N/K/HEAD_DIM 等）
3. 根据需要开启融合变体（注释中标注扩展点）
4. 参考 [hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md) 验证约束

## 相关页面
- [workflows/shape_router](../workflows/shape_router.md) — 选择哪个骨架的决策树
- [hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md) — 生成后的约束校验
- [ops/fused_ops](../ops/fused_ops.md) — 融合算子设计说明
