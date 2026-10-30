---
id: ops-transformer-moe-gating-top-k-softmax
title: CANN ops-transformer MoeGatingTopKSoftmax source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_gating_top_k_softmax/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeGatingTopKSoftmax 来源摘要

固定提交 `d28d8f9...`。支持 A2/A3 + 950。AIV-only。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `1fced6ff88174ec3bb6bf8eca301185f4c988685218a8ac907241dbc5c8cb38f` |
| `op_host/moe_gating_top_k_softmax_def.cpp` | `270d110c7a8b77da8b73d7fcddc5c057e80cd50660f4e753a6d27d1ad018379c` |
| `op_kernel/moe_gating_top_k_softmax.cpp` | `e00a1366c3ef28fb0fa4b030654a53209080b3400b8d37d509cef8b7747a437a` |

## 关键代码区域

- **功能**（README L16-33）：MoE gating softmax + topk。`softmax(x)` -> `topK` -> 输出 y/expertIdx/rowIdx。finished=True 时 expertIdx 填 num_expert。
- **算子定义**（def.cpp L23-59）：输入 x(FP16/FP32/BF16)/finished(OPTIONAL,BOOL)。输出 y/expert_idx/row_idx(INT32)。属性 k。AICore config：910b/910_93 + 950(opFile=..._apt)。
- **kernel dispatch**（`moe_gating_top_k_softmax.cpp` L49-113）：AIV-only。18 tiling key，3 模板族：E_K_FullLoad(0-5)/K_FullLoad(6-8)/Perf(9-17 按 ColRange <8/8-64/>64)。
- **约束**（README L93-96）：k≤1024；x 每维 < int32 max。

## 衍生引用

- `wiki/kernels/moe-gating-top-k-softmax.md`、`artifacts/moe-gating-top-k-softmax/`