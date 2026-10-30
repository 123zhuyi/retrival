---
id: ops-transformer-moe-token-permute
title: CANN ops-transformer MoeTokenPermute source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_token_permute/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenPermute 来源摘要

固定提交 `d28d8f9...`。仅 A2/A3，AIV-only。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `a06fd0b15378a19d68654ab18975711b2474f71929e93584a650b5fa841ef70a` |
| `op_host/moe_token_permute_def.cpp` | `f759de0a2ab05eb68d89e38ed520c581bd8c11aaae1a1b3158b1c96d720e8334` |
| `op_kernel/moe_token_permute.cpp` | `89b44b4604988e131e13ff59fc0f65995256e3815a17d0bdee247b87f908f8a0` |

## 关键代码区域

- **功能**（README L14-16）：MoE permute，根据 indices 将 tokens 广播并排序。
- **算子定义**（def.cpp L23-52）：输入 tokens(FP16/BF16/FP32/INT8)/indices(INT32/INT64)。输出 permute_tokens/sorted_indices(INT32)。属性 num_out_tokens(0)/padded_mode(false)。AICore config：910b/910_93。
- **kernel dispatch**（`moe_token_permute.cpp` L47-82）：AIV-only。8 tiling key（1-8）= one/multicore × normal/spilt_d × padded_false/true。3 阶段：Sort1(argSort indices) -> Sort2(argSort sortedFirst) -> IndexCopy。
- **约束**（README L100-113）：indices 元素 < 16777215；不支持 paddedMode=True；topK≤512；950 转调 moe_init_routing_v2。

## 衍生引用

- `wiki/kernels/moe-token-permute.md`、`artifacts/moe-token-permute/`