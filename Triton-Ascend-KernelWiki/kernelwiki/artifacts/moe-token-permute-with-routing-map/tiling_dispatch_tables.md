# moe_token_permute_with_routing_map - tiling 与 dispatch 分析

来源：`moe/moe_token_permute_with_routing_map/`（commit `d28d8f9…`）。MoE Token Permute With Routing Map 是 routingMap 驱动的 token 置换，三架构（A2/A3 + 950 arch35）。

## A. 架构分发

`def.cpp` L80-88 注册 AICore config：

| 架构 | SoC | config | kernel 入口 | 说明 |
|---|---|---|---|---|
| Atlas A2 训练/推理 | `ascend910b` L80 | `moe_token_permute_with_routing_map` | `moe_token_permute_with_routing_map.cpp` | membase 模板 |
| Atlas A3 训练/推理 | `ascend910_93` L81 | 同上 | 同上 | membase 模板 |
| Ascend 950PR/950DT | `ascend950` L88 | 同上（regbase） | 同上 | regbase + arch35 SIMD/SIMT gather |

> 三架构均支持。950 用 arch35 专属 `Gatherv2SimdTwoDim`/`Gatherv2SimtTwoDim`。支持 BF16/FP16/FP32 tokens + BOOL/INT8 routingMap。

## B. TilingKey 与 dispatch（kernel.cpp L99-154）

| TilingKey | 模式 | 说明 |
|---|---|---|
| 1/3/5/7 | 单核排序 | dropAndPad=false，单核 MoeSortOneCore |
| 2/4/6/8 | 多核排序 | dropAndPad=false，多核 MoeSortMultiCore/MoeSortMultiLastDimCore |
| 9/19 | 填充模式 | dropAndPad=true，仅 3510/arch35 |

## C. 核间切分

| 项 | 结论 |
|---|---|
| 逻辑轴 | tokens_num（row）、expert_num（col） |
| 多核排序 | 沿 token 维外切分到多核 + 归并（moe_mrgsort） |
| 单核排序 | 单核处理全部 |

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| maskedSelect | `tl.load(tokens + mask_nonzero)` | 间接 load |
| argSort | `tl.argsort` 或自定义排序 | 排序 |
| index_select | `tl.load(tokens + sorted_indices)` | 间接 load |
| 单核/多核排序 | `tl.program_id` 多核 + 单核 fallback | 切分策略 |
| arch35 SIMD/SIMT gather | `tl.load` 间接索引 | 不迁移 SIMD/SIMT 机制 |
| 归并排序 | `tl.argsort` | 不迁移归并算法 |
| SyncAll 多核同步 | 不迁移 | CANN 专有 |