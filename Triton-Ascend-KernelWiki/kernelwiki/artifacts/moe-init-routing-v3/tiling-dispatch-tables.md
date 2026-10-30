# MoeInitRoutingV3 - Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

MoE routing 计算（README.md），V2 的升级：增加动态量化、activeExpertRange、删除 expertTokensBeforeCapacity。6 步公式：

1. `sortedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)` - 排序
2. `expandedRowIdxOut[sortedRowIdx[i]] = i` - 位置映射
3. drop 模式：`expertTokensCountOrCumsumOut = Histogram(sortedExpertIdx)` - 直方图
4. 量化：
   - 动态量化无 scale：`dynamicQuantScale = row_max(abs(x))/127`，`quantResult = round(x/dynamicQuantScale)`
   - 动态量化有 scale：`dynamicQuantScale = row_max(abs(x*scale))/127`
   - 静态量化：用输入 scale/offset
5. `expandedXOut[i] = quantResult[sortedRowIdx[i] % NUM_ROWS]` - gather
6. `availableIdxNum = |{x ∈ expertIdx | expert_start ≤ x < expert_end}|` - active range 过滤

## B. 输入输出（def.cpp）

- **输入**：x（INT8/FP16/BF16/FLOAT；950 还支持 HIFLOAT8）、expert_idx（INT32）、scale（FLOAT，可选）、offset（FLOAT，可选）
- **输出**：expanded_x（INT8/FP16/BF16/FLOAT；950 还支持 FP8_E5M2/E4M3FN/HIFLOAT8）、expanded_row_idx（INT32）、expert_tokens_count_or_cumsum（INT64）、expanded_scale（FLOAT；950 支持 FP8_E8M0）
- **属性**：active_num、expert_capacity、expert_num、drop_pad_mode、expert_tokens_num_type、expert_tokens_num_flag、quant_mode、active_expert_range、row_idx_type

## C. 架构

- ascend910b/910_93（membase）
- ascend950（regbase，opFile=moe_init_routing_v3_apt，支持更多 dtype + FP8_E8M0 scale）

## D. 模板 dispatch（kernel L40-90）

`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only，g_coreType==AIC return）。~30 tiling key：

- **高性能 FullLoad**（2000000 系列）：UNQUANTIZED/STATIC_QUANT/DYNAMIC_QUANT × GATHER/SCATTER × scale 模式
- **非 FullLoad**：
  - 排序策略：SortOneCore（1000xxx）/ SortMultiCore（1100xxx）
  - 索引模式：GATHER / SCATTER
  - 量化模式：NonQuant / DynamicQuant（1020xxx）/ StaticQuant（1010xxx）
  - Drop/Pad：NoDrop / Drop（+100）
  - 有 Gather 前置：GatherSortOneCore（1200xxx）/ GatherSortMultiCore（1300xxx）