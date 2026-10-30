---
id: ops-transformer-moe-compute-expert-tokens
title: CANN ops-transformer MoeComputeExpertTokens source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_compute_expert_tokens/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeComputeExpertTokens 来源摘要

固定提交 `d28d8f9...`。全架构（A2/A3+950+Kirin）。AIV-only。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `a3d6274fe022835cc4520a78f337e386d6399c859755cef2d0e3aaeb5dc77a62` |
| `op_host/moe_compute_expert_tokens_def.cpp` | `ca7677e88db11e5f86d46b9a054ae924cc766faaacaee7711888a7f370a37b85` |
| `op_kernel/moe_compute_expert_tokens.cpp` | `d7763c6896e46b342281e10c0aee4019acbcc1f7fe98ac764319f7f2f7c0045a` |

## 关键代码区域

- **功能**（README L18-23）：MoE 通过二分查找找每个专家处理的最后一行位置，输出累积 token 数。
- **算子定义**（def.cpp L23-48）：输入 sorted_experts(INT32)。输出 total_rows_before_expert(INT32)。属性 num_experts。AICore config：910b/910_93/950/kirinX90/kirin9030。
- **kernel dispatch**（`moe_compute_expert_tokens.cpp` L21-45）：MIX_AIV_1_0（AIV-only）。3 tiling key（1001/1002/1003）= SS/M/L 按数据量。3 模板：`MoeComputeExpertTokensInt32{SS,M,L}`。
- **约束**（README L62-66）：sortedExperts shape < 2^24；numExperts >0 且 ≤2048。

## 衍生引用

- `wiki/kernels/moe-compute-expert-tokens.md`、`artifacts/moe-compute-expert-tokens/`