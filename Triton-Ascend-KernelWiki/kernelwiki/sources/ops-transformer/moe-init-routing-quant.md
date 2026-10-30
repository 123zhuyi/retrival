---
id: ops-transformer-moe-init-routing-quant
title: CANN ops-transformer MoeInitRoutingQuant source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_init_routing_quant/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeInitRoutingQuant 来源摘要

固定提交 `d28d8f9...`。仅 A2/A3，AIV-only。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `0b6fa60c4a82ead04f3b56db0155652bbd840884d10edf7a2f50ceaaec4c2a12` |
| `op_host/moe_init_routing_quant_def.cpp` | `93cb63f9c9f686a5e2c7a452c0f91e3beb0015b92225d374bc3a9bd94a2f0f76` |
| `op_kernel/moe_init_routing_quant.cpp` | `d9f4122bd40bc9ea225fdcb923f8e65b14c8883e836c84b191eb14e150d1b4e6` |
| `op_kernel/moe_quant_common.h` | `e5d9eb306059f85754c5f7268075ed4e5ab270abb1e8c4c15113c6b969ba5cb9` |
| `op_kernel/moe_quant_sort_multi_core.h` | `8b56fe6439bdfd440b6e6cdaa1f263a71fdf07d213a722be224a0ba3face25df` |
| `op_kernel/moe_quant_gather_out.h` | `741110eccc22860f5ede43041120265dd337aab359b53cc4494fa1512f71d1c3` |

## 关键代码区域

- **功能**（README L16-29）：MoE routing + 量化。3 阶段：Sort(expertIdx,rowIdx) -> SrcToDst(反向映射) -> GatherOut+Quant(INT8)。
- **算子定义**（def.cpp L23-60）：输入 x(FP16/BF16/FP32)/row_idx/expert_idx(INT32)。输出 expanded_x(INT8)/expanded_row_idx/expanded_expert_idx(INT32)。属性 active_num/scale/offset。AICore config：910b/910_93。
- **kernel dispatch**（`moe_init_routing_quant.cpp` L36-80）：AIV-only（`g_coreType==AIC return`）。4 tiling key（1/2 normal + 3/4 small_active_row × 1core/multicore）。3 阶段：MoeSort -> MoeSrcToDstOp -> MoeGatherOut。
- **约束**（README L109-111）：无特殊约束。

## 衍生引用

- `wiki/kernels/moe-init-routing-quant.md`、`artifacts/moe-init-routing-quant/`