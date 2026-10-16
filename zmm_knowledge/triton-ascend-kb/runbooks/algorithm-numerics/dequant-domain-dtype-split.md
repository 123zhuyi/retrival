---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.2"
title: 反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）
description: fp16 输入在 fp16 域反量化，bf16 输入在 fp32 域反量化后截断，对齐 cube L0C 行为
tags: [dequant, dtype_domain, fp16_domain, bf16_fp32, l0c]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-6 反量化域按 dtype 分流

【优化维度】：数值｜混合精度
【泛化层级】：条件性（迁移条件：权重反量化+matmul 融合；失效边界：纯 fp32 无须分流）

<原则：对齐 torch_npu op 的 cube L0C 行为--fp16 路径在 fp16 域反量化（精度足够且快），bf16 路径在 fp32 域反量化后截断（避免加法尾数损失再截断）。域不匹配导致 MERE 超阈值。与 [[fp32-reduction-accumulator]] 互补：本条是反量化域选择，N1 是累加精度。>

<骨架（占位名伪码）：
if x.dtype == fp16:
    w_dq = (w.to(f16) + off.to(f16)) * scale.to(f16)        # fp16 域
else:  # bf16
    w_dq = ((w.to(f32) + off.to(f32)) * scale.to(f32)).to(bf16)  # fp32 域后截断
>

**坏实践（反例）**：bf16 用 fp16 域或 fp16 用 fp32 域
```python
w_dq = (w.to(f16) + off.to(f16)) * scale.to(f16)  # bf16 输入走 fp16 域，加法尾数损失
```
后果：MERE 超阈值（fp16 9.77e-4 / bf16 7.81e-3），verify 失败。✅ 正解即按 dtype 分流。

**已知实例**：weight-quant-matmul W1（48/48 verify 验证）

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 同属优化知识，主题相关
- 相关主题: [计算优化: phase split 与 barrier 外提](../compute-pipeline/exp_kda_bwd_wy_structure_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关
- 相关主题: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../memory-sram/bf16-fused-dequant-blockk128.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 同属优化知识，主题相关
- 相关主题: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../memory-sram/exp_matmul_001.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 同属优化知识，主题相关
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
