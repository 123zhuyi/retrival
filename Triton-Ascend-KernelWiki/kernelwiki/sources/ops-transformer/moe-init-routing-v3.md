---
id: ops-transformer-moe-init-routing-v3
title: CANN ops-transformer MoeInitRoutingV3 source snapshot
type: source
repo: cann/ops-transformer
path: moe/moe_init_routing_v3/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeInitRoutingV3 来源摘要

仓库快照固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。`MoeInitRoutingV3` 是 MoE routing 计算算子，V2 的升级版：增加动态量化、activeExpertRange、删除 expertTokensBeforeCapacity。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_init_routing_v3/`

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `cec98117c595d5c14cbe99028170e7f77e1587af7b5531368e7680dddc75620e` | 算子总览、6 步公式、I/O、V2->V3 变更 |
| `op_host/moe_init_routing_v3_def.cpp` | `3b357f77d76427a09139ea058b74d087e2cdda709165afadc9549cca900027a5` | OpDef（2 套 AICore 配置）|
| `op_host/moe_init_routing_v3_tiling.h` | `9711e474b255ad92fecb92aa10423b0f071adcfaf3bdd1e3543ce2802395b68f` | TilingData 结构体 |
| `op_kernel/moe_init_routing_v3.cpp` | `168a4a1354b0a30505d6a0d93a56270add730c1cd131d312d68c7b62f7eea1df` | kernel 入口（~30 tiling key）|

## 3. 关键代码区域摘要

### 6 步公式（README.md）
1. `sortedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)`
2. `expandedRowIdxOut[sortedRowIdx[i]] = i`
3. drop 模式：`expertTokensCountOrCumsumOut = Histogram(sortedExpertIdx)`
4. 量化：动态 `row_max(abs(x))/127` + round；静态 `scale/offset`
5. `expandedXOut[i] = quantResult[sortedRowIdx[i] % NUM_ROWS]`
6. `availableIdxNum = |{x ∈ expertIdx | expert_start ≤ x < expert_end}|`

### 输入输出（def.cpp）
- 输入：x（INT8/FP16/BF16/FLOAT；950 +HIFLOAT8）、expert_idx（INT32）、scale（FLOAT，可选）、offset（FLOAT，可选）
- 输出：expanded_x（INT8/FP16/BF16/FLOAT；950 +FP8/HIFLOAT8）、expanded_row_idx（INT32）、expert_tokens_count_or_cumsum（INT64）、expanded_scale（FLOAT；950 +FP8_E8M0）
- 属性：active_num、expert_capacity、expert_num、drop_pad_mode、expert_tokens_num_type、expert_tokens_num_flag、quant_mode、active_expert_range、row_idx_type

### 架构（def.cpp L98-191）
ascend910b/910_93（membase）、ascend950（regbase，支持更多 dtype + FP8_E8M0 scale）

### Kernel dispatch（kernel L40-90）
`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only）。~30 tiling key：FullLoad 高性能（2000000 系列）+ SortOneCore/SortMultiCore × Gather/Scatter × NonQuant/DynamicQuant/StaticQuant × NoDrop/Drop。

## 4. 衍生引用
- `wiki/kernels/moe-init-routing-v3.md`
- `artifacts/moe-init-routing-v3/`