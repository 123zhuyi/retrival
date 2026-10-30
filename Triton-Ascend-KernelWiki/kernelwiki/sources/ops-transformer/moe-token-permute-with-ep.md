---
id: ops-transformer-moe-token-permute-with-ep
title: CANN ops-transformer MoeTokenPermuteWithEp source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_token_permute_with_ep/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenPermuteWithEp 来源摘要

固定提交 `d28d8f9...`。仅 A2/A3，AIV-only。`moe_token_permute` 的 EP（Expert Parallel）扩展版。

## 关键代码区域

- **功能**（README L16-48）：MoE permute with EP，根据 indices 将 tokens 和可选 probs 广播排序并按 range 切片。
- **算子定义**（def.cpp L23-67）：输入 tokens(FP16/BF16/FP32)/indices(INT32/INT64)/probs(OPTIONAL)。输出 permute_tokens/sorted_indices(INT32)/permute_probs。属性 range({0,0})/num_out_tokens(0)/padded_mode(false)。AICore config：910b/910_93。
- **kernel dispatch**（`moe_token_permute_with_ep.cpp` L51-86）：AIV-only。8 tiling key（1-8）= one/multicore × normal/spilt_d × padded_false/true。3 阶段：Sort1(argSort indices) -> Sort2(argSort sortedFirst) -> IndexCopy（tokens + probs 双输出，按 range 切片）。
- **约束**（README L128-133）：indices 元素 < 16777215；topK≤512；不支持 paddedMode=True；range 空时回退到 moe_token_permute。

## 衍生引用

- `wiki/kernels/moe-token-permute-with-ep.md`、`artifacts/moe-token-permute-with-ep/`