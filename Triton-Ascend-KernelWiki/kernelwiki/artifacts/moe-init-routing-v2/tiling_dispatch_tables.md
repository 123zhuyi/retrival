# moe_init_routing_v2 - tiling 与 dispatch 分析

来源：`moe/moe_init_routing_v2/`。MoE 路由 v2。

## A. 架构与模板分发

`def.cpp` L60-146：ascend910b（A2）、ascend910_93（A3）、ascend310p、ascend950（950PR/950DT）。**四架构**（v1 仅三架构，无 310P）。

`MIX_AIV_1_0`（纯 Vector，`moe_init_routing_v2.cpp`）。tiling key 分派（类比 v1 的 5 key）：0=full load / 1-2=单/多核sort+正常gather / 3-4=单/多核sort+小活跃行gather。

## B. 与 v1 的关键差异

| 特性 | v1 | v2 |
|---|---|---|
| rowIdx 输入 | 有 | **无** |
| activeNum | 必选 | 可选（默认0） |
| expert_capacity | 无 | **新增** |
| expert_num | 无 | **新增** |
| drop_pad_mode | 无 | **新增** |
| expert_tokens_count_or_cumsum 输出 | 无 | **新增**（可选） |
| expert_tokens_before_capacity 输出 | 无 | **新增**（可选） |
| ascend310p 支持 | 无 | **新增** |
| 架构数量 | 3（A2/A3/950） | **4（A2/A3/950/310P）** |

## C. Triton-Ascend 抽象映射

| CANN | Triton | 说明 |
|---|---|---|
| sort(expertIdx) | `tl.sort` / scatter | 按 expert 排序 |
| srcToDstWithCapacity | scatter + capacity 裁剪 | 支持容量上限 |
| expertTokenOut | 计数输出 | 每 expert token 数 |
| gather expandedX | `tl.load(x + idx*cols)` | 按索引 gather |
| drop_pad_mode | mask 或截断 | 超容量丢弃/填充 |
| MIX_AIV_1_0 | 纯 vector kernel | 无需 cube |