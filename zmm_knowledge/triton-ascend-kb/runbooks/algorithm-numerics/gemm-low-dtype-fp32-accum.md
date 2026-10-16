---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.2"
title: GEMM 操作数保持 low_dtype，fp32 累加
description: tl.dot 用 low_dtype 操作数（cube 原生 fp16×fp16->fp32），fp32 累加器累加，禁止操作数先升 fp32
tags: [gemm, low_dtype, fp32_accum, cube_native, dot]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-7 GEMM low_dtype + fp32 累加

【优化维度】：数值｜混合精度
【泛化层级】：条件性（迁移条件：量化权重的 matmul；失效边界：纯 fp32 matmul 无 low_dtype）

<原则：cube 原生 fp16×fp16->fp32 / bf16×bf16->fp32，low_dtype 值在 fp32 精确可表，乘积精确，fp32 累加 == 标杆 fp32 split-K。把操作数先 to(f32) 再 dot 是非 cube 原生模式，慢且无精度收益。与 [[dequant-domain-dtype-split]] 配套。>

<骨架（占位名伪码）：
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
acc += tl.dot(x_low, w_dq_low)   # low_dtype 操作数，cube 原生 -> fp32 累加
# 禁止：acc += tl.dot(x.to(f32), w.to(f32))  # 非 cube 原生，慢
>

**坏实践（反例）**：操作数先升 fp32 再 dot
```python
acc += tl.dot(x.to(tl.float32), w_dq.to(tl.float32))  # 非 cube 原生模式
```
后果：非 cube 原生模式，性能下降且无精度收益。✅ 正解即 low_dtype 操作数 + fp32 累加。

**已知实例**：weight-quant-matmul W2（== 标杆 fp32 split-K）

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — FwdH 全 fp32 累加实践与该 low_dtype/fp32 累加契约一致
- 开发指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 gemm
- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 gemm
- 相关主题: [Gemm Basic Skeleton](../../reference/skeleton-gemm_basic.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: gemm based histogram](../compute-pipeline/exp_digit_hist_001.md) — 同属优化知识，主题相关
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](../compute-pipeline/exp_kda_bwd_scalar_loop_to_dot_001.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 gemm
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](../compute-pipeline/scalar-loop-to-dot.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 同属优化知识，主题相关
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 同属优化知识，主题相关
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
