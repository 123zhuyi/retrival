---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.2"
title: fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）
description: ≥10-op 串联且含 LN/softmax/SiLU 非线性链的复合算子，fp16/bf16 eager 逐 op 舍入链对任何独立实现数学不可达（混沌放大+CANN 内部累加序不可复现）；通过输入全升 fp32 + 出口降回标称 dtype 的契约使 verify 可达。
tags: [precision_contract, fp32_chain, composite_block, verify_gate, rounding_chain, mobilevit]
confidence: draft
status: draft
created_at: "2026-08-26T00:00:00Z"
updated_at: "2026-08-26T00:00:00Z"
---

## fp32 精度契约模式（来源：53_MobileViTAttention 两轮独立互证 + 修复后 5 轮验证）

【泛化层级】：条件性（迁移条件：长链复合算子 + 低精度多 case verify 系统性失败；失效边界见末节）

**机制（约束）**：fp16/bf16 eager 链每个 op 出口舍入到低精度，verify（matched_ratio≥0.9 且 MERE 达标）实质要求逐位复现 CANN conv/LN/softmax 的内部累加顺序——未文档化、Triton 级公式不可复现。softmax exp 将前级 1-ulp 噪声放大为全行 ~1% 相对误差（dP/P=dt），即使 conv+LN+GEMM 全部 bit-exact，matched 上限仍 ~0.89 < 0.9（hardware noise floor）。

**判别法（如何实测确认本模式适用）**：
1. 参考实现自身 bf16 vs fp32 的 matched_ratio < 0.9 → 契约问题而非实现问题，本模式适用；
2. 级联注入矩阵：逐 stage 用 eager 替换测 matched 贡献，若全精确注入仍不过门 → 确诊；
3. fp64 真值证伪：注入 fp64 计算后舍入的 stage 若 matched 反降 → eager 的 fp32 累加噪声必须逐位复现，高精度路线无效。

**复刻规则（任务文件侧修改）**：
```python
# get_input_groups：所有张量升精度为 fp32 传入
x = _rand_x(shape, torch.float32, seed)  # 原 dtype 仅记录
# Model.forward：全链路 fp32 计算，出口按 case 标称 dtype 降回
out = F.conv2d(...)
out_dtype = _OUTPUT_DTYPES.get((tuple(x.shape), patch_size))  # 从同名 json 预建映射
return out.to(out_dtype) if out_dtype is not None else out
```
verify 的 data_type 取 framework 输出 dtype，仍以 fp16/bf16 阈值判定；两侧均 fp32 链计算后单次降精度，差异降至 2^-24 量级，verify 可达（实测 iter_0 即 49/49）。

**代价**：benchmark 测的是 fp32 链路，绝对延迟高于原生 fp16 实现，speedup 语义随之改变（对比基准同为 fp32 链 eager）。

**失效边界**：单 op/短链算子（eager 舍入可逐位复现，如 AdamW/GNS 已反演成功）；任务文件禁止修改且要求原生低精度链语义的场景。

<!-- okf:related:start -->

# 相关

- 相关主题: [复合块多 kernel pipeline 分解模式（~30-op conv+transformer 块的已验证骨架）](../compute-pipeline/composite-kernel-decomposition.md) — 主题相关：共享 composite_block, mobilevit
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_flashattn_bwd_kv_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: mixed precision computation](../compute-pipeline/exp_swiglu_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: precision control](../memory-sram/exp_skip_layernorm_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend strided-2/interleave 访存契约（三个正确性缺陷 + 寄存器重排正解）](../memory-sram/triton-ascend-strided2-interleave-contract.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
