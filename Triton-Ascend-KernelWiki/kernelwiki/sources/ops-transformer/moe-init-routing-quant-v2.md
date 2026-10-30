---
id: ops-transformer-moe-init-routing-quant-v2
title: CANN ops-transformer MoeInitRoutingQuantV2 source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_init_routing_quant_v2/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeInitRoutingQuantV2 来源摘要

固定提交 `d28d8f9...`。支持 A2/A3 + 950(arch35)。AIV-only。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `ef5b6f52a0eb542fffb90dd7f9a7ead352113c6e2287e98dc559fc2b14b9e90f` |
| `op_host/moe_init_routing_quant_v2_def.cpp` | `644fd0747e0fa64cd0d2012ef6ff97b3e17738c0a0f00ad832e9cce5a250c5b1` |
| `op_kernel/moe_init_routing_quant_v2.cpp` | `5707a400460155e8f8713e9081b5105f9b023a825035b29f0d613a5300b9beac` |
| `op_kernel/moe_init_routing_quant_v2_apt.cpp` | `36a5153673d015d95b6c82c3a7c4122c87110378ea28b7596607b39dc2e00405` |
| `op_kernel/moe_v2_fullload_quant.h` | `933d68d00c35dd8c0e8acfc11fad5398334a6a2a24b11bb647d0e286f181d4c2` |
| `op_kernel/moe_v2_gather_quant.h` | `0a848ac2515e40ce8bafdda253d5290a2d207bc3666e0e79e6e497e7b977a1a8` |

## 关键代码区域

- **功能**（README L14-24）：MoE routing V2。新增 drop/dropless 模式、动态 quant、删除 rowIdx、expertTokens 统计。
- **算子定义**（def.cpp L23-91）：输入 x(FP16/BF16/FP32)/expert_idx(INT32)/scale/offset(OPTIONAL)。输出 expanded_x(INT8/INT4)/expanded_row_idx/expert_tokens_count_or_cumsum/expert_tokens_before_capacity/dynamic_quant_scale(OPTIONAL)。属性 active_num/expert_capacity/expert_num/drop_pad_mode/expert_tokens_count_or_cumsum_flag/expert_tokens_before_capacity_flag/quant_mode。AICore config：910b/910_93 + 950(opFile=..._apt)。
- **kernel dispatch**（`moe_init_routing_quant_v2.cpp` L43-147）：MIX_AIV_1_0（AIV-only）。3 档 tiling key：20000(静态fullload)/21000(动态fullload)/10000-11110(多阶段)。arch35 入口（`_apt.cpp`）有 regbase/simt 分支。
- **公式**（README L25-83）：6 步 Sort->SrcToDst->ExpertToken->Quant(静态/动态)->Gather->Drop容量。

## 衍生引用

- `wiki/kernels/moe-init-routing-quant-v2.md`、`artifacts/moe-init-routing-quant-v2/`