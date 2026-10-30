# TopKPQDistanceV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入pq_distance的每个batch行，取该行element_num个元素中最大或最小的k个值及其列索引，分别输出到topk_distance与topk_index。grouped_extreme_distance提供每组极值，仅用于剪枝加速，不改变计算结果。

- 计算公式：将某个batch行的元素按升序排列记为$s_0 \le s_1 \le \dots \le s_{n-1}$，其中$n$为element_num。按order取极值，$i=0,1,\dots,k-1$：
  - order为“ASC”时取最小的k个并升序输出，$\mathrm{topk\_distance}[i]=s_i$；
  - order为“DES”时取最大的k个并降序输出，$\mathrm{topk\_distance}[i]=s_{n-1-i}$。

  $\mathrm{topk\_index}[i]$为对应元素在该行pq_distance中的原始列索引。

- 计算过程：借助分组极值剪枝快速得到上述结果。
  1. 依据grouped_extreme_distance选出最有可能包含前k个极值的若干分组，并按优劣排序。
  2. 用这些分组内的元素初始化大小为k的堆，再遍历剩余分组，利用分组极值剪枝，只在可能更优时才更新堆。
  3. 将堆中元素依次弹出，得到有序的topk_distance与topk_index。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>pq_distance</td>
      <td>输入</td>
      <td>距离张量，2维，shape为`[batch, element_num]`。</td>
      <td>FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grouped_extreme_distance</td>
      <td>输入</td>
      <td>每个分组的极值，2维，shape为`[batch, element_num / group_size]`。order为“ASC”时取每组最小值，为“DES”时取每组最大值。数据类型需与pq_distance一致。</td>
      <td>FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_distance</td>
      <td>输出</td>
      <td>最后一维上k个最大或最小的元素值，shape为`[batch, k]`。数据类型与pq_distance一致。</td>
      <td>FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_index</td>
      <td>输出</td>
      <td>topk_distance中各元素在pq_distance最后一维中的索引，shape为`[batch, k]`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>order</td>
      <td>属性</td>
      <td>可选属性，表示排序方式，取值为“ASC”或“DES”。默认值为“ASC”。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>k</td>
      <td>属性</td>
      <td>必选属性，表示取最大或最小元素的个数。取值需大于0且不大于element_num。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_size</td>
      <td>属性</td>
      <td>必选属性，表示求分组极值时的分组大小。取值需大于0，且element_num需为group_size的整数倍。</td>
      <td>INT</td>
      <td>-</td>
    </tr>

  </tbody></table>

## 约束说明

- grouped_extreme_distance须与order取值匹配：order为“ASC”时应为每组最小值，为“DES”时应为每组最大值，否则剪枝可能得不到预期结果。
- 当存在距离值相等的元素时，排序为非稳定排序，相等元素之间的先后顺序不确定。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_top_k_pq_distance_v2](./examples/test_geir_top_k_pq_distance_v2.cpp)   | 通过[算子IR](./op_graph/top_k_pq_distance_v2_proto.h)构图方式调用TopKPQDistanceV2算子。 |
