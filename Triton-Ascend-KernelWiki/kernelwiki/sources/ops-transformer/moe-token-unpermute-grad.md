---
id: ops-transformer-moe-token-unpermute-grad
title: CANN ops-transformer MoeTokenUnpermuteGrad source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_token_unpermute_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenUnpermuteGrad 来源摘要

固定提交 `d28d8f9...`。仅 A2/A3，AIV-only。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `977d0415b0ccc9d18f07ad822540519911fe37c8c5c6af2e7fc42f22423a811f` |
| `op_host/moe_token_unpermute_grad_def.cpp` | `bd2c0a21afcf7f5b88d6647b72d28cec4a1659bc2ebfa38d8ff394dd3ec7b923` |
| `op_kernel/moe_token_unpermute_grad.cpp` | `03e6ea18a929e265522a6fbd03eee039fe9f1fb8b0d588f6e8cd7ef433574ba9` |

## 关键代码区域

- **功能**（README L16-45）：`aclnnMoeTokenUnpermute` 的反向传播。计算 permutedTokensGrad（scatter 加权）和 probsGrad（topK 求和）。
- **算子定义**（def.cpp L33-77）：输入 permuted_tokens/unpermuted_tokens_grad/sorted_indices/probs(OPTIONAL)（FP16/BF16/FP32）。输出 permuted_tokens_grad/probs_grad。属性 padded_mode/restore_shape。AICore config：910b/910_93。
- **kernel dispatch**（`moe_token_unpermute_grad.cpp` L30-46）：4 tiling key（0/1/10/11）= padded_mode × haveProbs。2 模板：`MoeTokenUnpermuteGradProbNone`（无 probs）/ `MoeTokenUnpermuteGradProbNotNone`（有 probs）。
- **约束**（README L118-129）：topK≤512；950 转调 finalize_routing_v2_grad。

## 衍生引用

- `wiki/kernels/moe-token-unpermute-grad.md`、`artifacts/moe-token-unpermute-grad/`