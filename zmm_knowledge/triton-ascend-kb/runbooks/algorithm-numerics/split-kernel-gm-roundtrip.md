---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.3"
title: 编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip
description: 参考实现物化中间 dtype 舍入时，Triton 融合会跳过舍入导致逐 bit 不匹配，必须拆 kernel 强制 GM store+load
tags: [gm_roundtrip, dtype_rounding, fusion_split, bit_exact]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-4 拆分 kernel 强制 GM round-trip

【优化维度】：数值｜数值稳定
【泛化层级】：条件性（迁移条件：参考实现（PyTorch C++）在每个子算子物化中间结果为输入 dtype；失效边界：fp32 输入融合后仍 fp32 无舍入损失，不必拆分）

<原则：bf16/fp16 下若参考实现在每个子算子物化中间结果（舍入到输入 dtype），Triton 编译器融合为单一 fp32 计算会跳过舍入，导致逐 bit 不匹配。必须拆分为多个独立 kernel，每个 kernel 通过 GM store+load 往返强制中间 dtype 舍入。与 [[fp32-reduction-accumulator]] 互补：本条是强制降精度舍入，N1 是强制升精度累加。>

<骨架（占位名伪码）：
# 参考实现：tanh(x/30)*30 在 bf16 下每步舍入
# ❌ 融合：result = tl.tanh(x/30.0)*30.0  # fp32 贯通，跳过 bf16 舍入
# ✅ 拆分：
kernel1[grid](x, tmp, ...)        # tmp = (x/30).to(bf16)  GM store
kernel2[grid](tmp, tmp2, ...)     # tmp2 = tanh(tmp).to(bf16)  GM store
kernel3[grid](tmp2, out, ...)     # out = tmp2*30  GM store
>

**坏实践（反例）**：融合表达式跳过中间舍入
```python
result = tl.tanh(x / 30.0) * 30.0   # bf16 输入但 fp32 贯通，无中间 bf16 舍入
```
后果：逐 bit 不匹配 PyTorch 参考（参考实现每步舍入到 bf16），verify 失败。✅ 正解即拆 kernel 强制 GM round-trip。

**已知实例**：transformer-inference T5（AttentionSoftmaxWithSoftcappingAndDropout 的 tanh(x/30)*30 拆 div/tanh/mul 三独立 kernel）、embedding-mlp-fusion L1.1

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 同属优化知识，主题相关
- 相关主题: [计算优化: phase split 与 barrier 外提](../compute-pipeline/exp_kda_bwd_wy_structure_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 同属优化知识，主题相关
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
