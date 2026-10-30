# InplaceTopKDistance

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

- 算子功能：将一组已排序的距离数据（topk_pq_distance）与一组新的距离数据（pq_distance）合并重排后取前k个，并将结果原地写回topk_pq_distance、topk_pq_index、topk_pq_ivf。其中k为topk_pq_distance的元素个数。该算子无输出，计算结果原地更新到上述输入中。

- 计算公式：将topk_pq_distance与pq_distance的全部元素按距离值升序排列记为$s_0 \le s_1 \le \dots \le s_{m-1}$，其中$m$为两者元素个数之和，$k$为topk_pq_distance的元素个数。按order写回前k个，$i=0,1,\dots,k-1$：
  - order为“asc”时取最小的k个并升序写回，$\mathrm{topk\_pq\_distance}[i]=s_i$；
  - order取其他值时取最大的k个并降序写回，$\mathrm{topk\_pq\_distance}[i]=s_{m-1-i}$。

  每个元素的索引与桶编号（topk_pq_index、topk_pq_ivf）随其距离值一同移动写回。

- 计算过程：
  1. 将topk_pq_distance的每个元素与其对应的topk_pq_index、topk_pq_ivf组成三元组；将pq_distance的每个元素与其对应的pq_index以及标量pq_ivf组成三元组。
  2. 将两组三元组合并后，按距离值升序排序。
  3. 按上述公式取前k个或后k个三元组写回。

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
      <td>topk_pq_distance</td>
      <td>输入</td>
      <td>已排序的距离张量，计算后被原地更新。其元素个数即为取前k个数据的k。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_pq_index</td>
      <td>输入</td>
      <td>与topk_pq_distance对应的索引，计算后被原地更新。元素个数与topk_pq_distance一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topk_pq_ivf</td>
      <td>输入</td>
      <td>与topk_pq_distance对应的桶编号，计算后被原地更新。元素个数与topk_pq_distance一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pq_distance</td>
      <td>输入</td>
      <td>新增的距离张量，将与topk_pq_distance合并重排。数据类型需与topk_pq_distance一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pq_index</td>
      <td>输入</td>
      <td>与pq_distance对应的索引。元素个数与pq_distance一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pq_ivf</td>
      <td>输入</td>
      <td>与pq_distance对应的桶编号，为标量或含1个元素的一维Tensor，pq_distance的所有元素共用该桶编号。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>order</td>
      <td>属性</td>
      <td>可选属性，表示topk_pq_distance的排序方式。取值为“asc”时结果按升序写回，取其他值时按降序写回。默认值为“asc”。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## 约束说明

- 当存在距离值相等的元素时，排序为非稳定排序，相等元素之间的先后顺序不确定。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_inplace_top_k_distance](./examples/test_geir_inplace_top_k_distance.cpp)   | 通过[算子IR](./op_graph/inplace_top_k_distance_proto.h)构图方式调用InplaceTopKDistance算子。该算子无输出，样例中以图target节点方式驱动执行。 |
