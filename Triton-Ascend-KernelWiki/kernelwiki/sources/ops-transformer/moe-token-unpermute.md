---
id: ops-transformer-moe-token-unpermute
title: CANN ops-transformer MoeTokenUnpermute source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_token_unpermute/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenUnpermute 来源摘要

固定提交 `d28d8f9...`。支持 A2/A3 + 310P + Kirin。AIV-only。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `dc9e84e42d2251d4a3643ea1f8442fe5313a93f0f8d232742381f79ad3227b25` |
| `op_host/moe_token_unpermute_def.cpp` | `abdc50111f6158a43feeced12399c116ba61e027047b729553f5f9d7328d515e` |
| `op_kernel/moe_token_unpermute.cpp` | `2e2de67a1169672496389a81e47d8b6b22079032196cb943bef1241f4ef53313` |

## 关键代码区域

- **功能**（README L18-47）：MoE token unpermute，按 sortedIndices 从 permutedTokens gather，若有 probs 相乘，再按 topK 累加还原。
- **算子定义**（def.cpp L23-118）：输入 permuted_tokens(FP16/BF16/FP32)/sorted_indices(INT32)/probs(OPTIONAL)。输出 unpermuted_tokens。属性 padded_mode(false)/restore_shape。AICore config：910b/910_93 + 310p/kirinX90/kirin9030(仅 FP16/FP32)。
- **kernel dispatch**（`moe_token_unpermute.cpp` L33-87）：~16 tiling key，dtype(BF16/FP16/FP32)×haveProbs×mix mode（probs dtype ≠ token dtype）。`KernelMoeTokenUnpermute<T1,T2,T3,haveProbs>`。
- **约束**（README L106-119）：topK≤512；950 转调 finalize_routing_v2；310P 仅 FP16/FP32，hiddensize 128 倍数≤10240。

## 衍生引用

- `wiki/kernels/moe-token-unpermute.md`、`artifacts/moe-token-unpermute/`