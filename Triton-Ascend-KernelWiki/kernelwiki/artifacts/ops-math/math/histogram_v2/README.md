# HistogramV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：计算张量直方图。
- 计算公式：以min和max作为统计上下限，在min和max之间划出等宽的数量为bins的区间，统计张量x中元素在各个区间的数量。如果min和max相等，则使用张量中所有元素的最小值和最大值作为统计的上下限。小于min和大于max的元素不会被统计。

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
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入张量x。</td>
      <td>FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>输入</td>
      <td>公式中的输入张量min，只含有1个元素。</td>
      <td>FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>输入</td>
      <td>公式中的输入张量max，只含有1个元素。</td>
      <td>FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，直方图的计算结果。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bins</td>
      <td>可选属性</td>
      <td><ul><li>表示被划分的区间数量。</li><li>默认值为100。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_histogram_v2](./examples/test_geir_histogram_v2.cpp)   | 通过[算子IR](./op_graph/histogram_v2_proto.h)构图方式调用HistogramV2算子。 |
