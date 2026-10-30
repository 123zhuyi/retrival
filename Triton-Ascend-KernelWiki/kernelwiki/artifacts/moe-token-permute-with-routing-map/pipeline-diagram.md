# moe_token_permute_with_routing_map - 流水编排分析

来源：`op_kernel/moe_token_permute_with_routing_map.cpp`（154 行）+ 13 个 kernel 头文件。MoE Token Permute With Routing Map 是 routingMap 驱动的 token 置换，核心是 maskedSelect + argSort + index_copy。

## 1. 核心计算公式

### dropAndPad=false（topK 模式）

```
expertIndex = arange(tokens_num).expand(expert_num, -1)
sortedIndicesFirst = expertIndex.maskedSelect(routingMap.T)    # 按 routingMap 选 token
sortedIndicesOut = argSort(sortedIndicesFirst)                  # 排序
topK = numOutTokens // tokens_num
permuteTokens[sortedIndicesOut[i]] = tokens[i // topK]          # 置换
permuteProbsOut = probs.T.maskedSelect(routingMap.T)             # probs 置换
```

### dropAndPad=true（capacity 模式）

```
capacity = numOutTokens // expert_num
sortedIndicesOut = argsort(routingMap.T, dim=-1)[:, :capacity]   # 排序后取前 capacity
permutedTokensOut = tokens.index_select(0, sortedIndicesOut)     # 置换
# probs: 按 (expert_num, capacity) 二维索引 gather
```

## 2. 单核/多核流程

```
1. KernelMaskedSelectV3: routingMap.T -> sortedIndicesFirst（maskedSelect）
2. MoeSort（OneCore/MultiCore/MultiLastDimCore）: sortedIndicesFirst -> sortedIndicesOut（argSort）
   - 多核：分核排序 + moe_mrgsort 归并
3. MoeindexCopyOp/MoeindexCopySplitDOp: tokens.index_select(sortedIndicesOut) -> permuteTokens
4. MoePermuteProb: probs 置换 -> permuteProbsOut
5. arch35: Gatherv2Simd/SimtTwoDim（二维 gather 优化）
```

## 3. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| maskedSelect | `KernelMaskedSelectV3` | `tl.load(tokens + mask_nonzero)` | 是（语义） |
| argSort | `MoeSort*` + `MoeMrgsort` 归并 | `tl.argsort` | 是（语义，归并算法 needs_verification） |
| index_select | `MoeindexCopyOp` | `tl.load(tokens + sorted_indices)` | 是（语义） |
| probs 置换 | `MoePermuteProb` | 同 tokens 路径 | 是（语义） |
| arch35 SIMD/SIMT gather | `Gatherv2Simd/SimtTwoDim` | `tl.load` 间接索引 | 是（语义，SIMD/SIMT 机制否） |
| 多核归并 SyncAll | CrossCoreSync | 不迁移 | 否（机制） |
| MTE 搬运 | DataCopyPad | `tl.load`/`tl.store` | 否（机制） |