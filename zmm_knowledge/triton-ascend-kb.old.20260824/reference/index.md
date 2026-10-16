---
kind: index
type: section_index
title: reference/ 参考卡
description: Triton-Ascend 官方文档、API、算子类别、代码骨架的结构化蒸馏卡与 legacy SPEC。
---

# reference/ 参考卡

## API 与硬件

- [Triton-Ascend API Cheatsheet](triton-api-cheatsheet.md) — Triton-Ascend API 速查
- [Ascend 910B 硬件概述](ascend-910b.md) — Atlas A2 训练系列 AI Core / 存储 / 流水概述
- [Ascend 优化模式](ascend-patterns.md) — 常见 Triton-Ascend 优化模式与反模式

## 算子类设计

- [Attention](op-class-attention.md) — 注意力类算子设计要点
- [Convolution](op-class-convolution.md) — 卷积类算子设计要点
- [Elementwise](op-class-elementwise.md) — 逐元算子设计要点
- [Embedding](op-class-embedding.md) — Embedding 类算子设计要点
- [Fused Ops](op-class-fused_ops.md) — 融合算子设计要点
- [Gather/Scatter](op-class-gather_scatter.md) — Gather/Scatter 类算子设计要点
- [Loss](op-class-loss.md) — Loss 类算子设计要点
- [MatMul](op-class-matmul.md) — 矩阵乘法类算子设计要点
- [Normalization](op-class-normalization.md) — 归一化类算子设计要点
- [Reduction](op-class-reduction.md) — 归约类算子设计要点
- [RoPE](op-class-rope.md) — 旋转位置编码设计要点
- [Softmax](op-class-softmax.md) — Softmax 类算子设计要点
- [Solve Tril](op-class-solve_tril.md) — 下三角求解类算子设计要点
- [SSM](op-class-ssm.md) — 状态空间模型算子设计要点

## 代码骨架

- [Flash Attention Causal Skeleton](skeleton-flash_attention_causal.md) — Causal FlashAttention 骨架
- [GEMM Basic Skeleton](skeleton-gemm_basic.md) — 基础 GEMM 骨架
- [MoE Dispatch Skeleton](skeleton-moe_dispatch.md) — MoE dispatch 骨架
- [RMSNorm Fused Skeleton](skeleton-rmsnorm_fused.md) — 融合 RMSNorm 骨架

## Legacy 规范

- [SPEC-Retrieve-legacy](SPEC-Retrieve-legacy.md) — 原 triton-knowledge-retrieval 缓存/检索协议
- [SPEC-Knowledge-Unit](SPEC-Knowledge-Unit.md) — 原知识单元格式规范
