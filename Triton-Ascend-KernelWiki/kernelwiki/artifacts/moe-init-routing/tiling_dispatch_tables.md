# moe_init_routing - tiling 与 dispatch 分析

来源：`moe/moe_init_routing/`（commit `d28d8f9...`）。MoE 路由算子。

## A. 架构与模板分发

`def.cpp` L63-71 注册 `ascend910b`（A2）、`ascend910_93`（A3）、`ascend950`（950PR/950DT）。产品支持（README L5-12）：A2 √、A3 √、950PR/950DT √。与之前入库的 attention 算子不同，本算子**三架构均支持**（attention 家族多数仅 A2/A3）。

任务类型：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)`（`moe_init_routing.cpp` L31），**纯 Vector 核**，无需 Cube。`MIX_AIV_1_0` = AIV 全占。

tiling key 分派（`moe_init_routing.cpp` L48-99）：

| key | 含义 | 组件 |
|---|---|---|
| 0 | 全量加载（full load） | MoeFullLoad（含排序+映射+gather 全流程） |
| 1 | 单核排序 + 正常 gather | MoeSortOneCore → MoeSrcToDstOp → MoeGatherOut |
| 2 | 多核排序 + 正常 gather | MoeSortMultiCore → MoeSrcToDstOp → MoeGatherOut |
| 3 | 单核排序 + 小活跃行 gather | MoeSortOneCore → MoeSrcToDstOp → MoeGatherOutSmallActiveRow |
| 4 | 多核排序 + 小活跃行 gather | MoeSortMultiCore → MoeSrcToDstOp → MoeGatherOutSmallActiveRow |

5 个 tiling key。无 dtype 分支（模板参数 `DTYPE_X` 由编译期 `ORIG_DTYPE_X` 确定）。

## B. 核间切分

- `MoeInitRoutingTilingData`（tiling.h L72-83）：`coreNum`、`n`（token 数）、`cols`（特征维）、`k`（topK 数）。
- 5 个子 tiling struct：VBSCompute（vector base sort）、VMSMiddle（vector merge sort）、SortOut、GatherOut（含 srcToDst）。
- 切分策略：按 N（token 数）和 K（topK 数）在 AIV 核间分配。

## C. 流水

三段流水（tiling key 1-4）：
1. **Sort**：对 `(expertIdx, rowIdx)` 按 expert 排序（单核/多核 merge sort）
2. **SrcToDst**：建立 `expandedRowIdx` 映射（sortedRowIdx[i] → i）
3. **Gather**：按 expandedRowIdx 从 `x` gather 出 `expandedX`

tiling key 0（full load）将三段合并为单次操作。

## D. Triton-Ascend 抽象映射

| CANN | Triton | 说明 |
|---|---|---|
| sort(expertIdx, rowIdx) | `tl.argsort` 或基于 expert 的 scatter | 按 expert 分组排序 |
| expandedRowIdx 映射 | `tl.arange` + scatter | 逆映射 |
| gather expandedX | `tl.load(x + idx * cols)` | 按索引 gather |
| activeNum 控制行数 | mask 或 slice | 仅前 activeNum 行有效 |
| core id（N×K 切分） | `tl.program_id` + `grid` | 按 token/expert 切分 |
| MIX_AIV_1_0 | 纯 vector kernel | 无需 cube |