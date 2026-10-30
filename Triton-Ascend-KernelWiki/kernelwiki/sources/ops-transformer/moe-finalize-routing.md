---
id: ops-transformer-moe-finalize-routing
title: CANN ops-transformer MoeFinalizeRouting source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_finalize_routing/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeFinalizeRouting 来源摘要

固定提交 `d28d8f9...`。支持 A2/A3 + Kirin。BF16+FP 双谱系 19 模板。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `4b4a77711d30e868a26bc94979fcb8de81045a4efcd3add69deb4dadd55c3c9d` |
| `op_host/moe_finalize_routing_def.cpp` | `fca675ec1e6ed1abdce5672216544871fabb30e4dacc99f14e74057d8bb1c8f8` |
| `op_kernel/moe_finalize_routing.cpp` | `efe91ee712995142b04c400dcdd0a735e485d5e305af5eda4fe31d18d0342dd5` |
| `op_kernel/moe_finalize_routing_common.h` | `860044e030eecdae9497d30f76e23854920758531fb07ae3f33904c70a83546b` |

## 关键代码区域

- **功能**（README L18-27）：MoE FFN 输出合并。`out = x1 + x2 + Σ scales·(expandedX[expandedRowIdx] + bias[expertId])`。
- **算子定义**（def.cpp L22-84）：输入 expanded_x/x1/x2(opt)/bias/scales/expanded_row_idx/expanded_expert_idx（REQUIRED，FP16/FP32/BF16，ND）。输出 y。AICore config：910b/910_93 + Kirin(仅 FP16/FLOAT)。
- **kernel dispatch**（`moe_finalize_routing.cpp` L43-142）：19 tiling key(20000-20018)，dtype×变体（CutK/Cuth/K2/K4/Db/DbAllBias/AllBias/CuthK2Optimized）。`BUFFER_NUM=1`，`PARALLEL_NUM=2`（common.h L26-27）。
- **约束**（README L101-103）：无特殊约束。

## 衍生引用

- `wiki/kernels/moe-finalize-routing.md`、`artifacts/moe-finalize-routing/`