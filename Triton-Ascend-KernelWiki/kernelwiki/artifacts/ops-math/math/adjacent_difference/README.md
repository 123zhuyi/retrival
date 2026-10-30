# AdjacentDifference

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：比较输入张量相邻元素的差异，若相邻元素相同，返回0，否则返回1。

- 计算公式：

$$
out_i = {(x_i == x_j ?\ 0 : 1)}.to(y_dtype) \ j = i - 1
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>待进行AdjacentDifference计算的入参，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>输入</td>
      <td>属性，指定输出类型，公式中的y_dtype。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行AdjacentDifference计算的出参，公式中的out。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

- 无。

## 调用说明

| 调用方式 | 样例代码 |   说明    |
| :-------- | :---------- | :------------------------ |
|  图模式  |[test_geir_adjacent_difference](examples/test_geir_adjacent_difference.cpp) | 通过[算子IR](./op_graph/adjacent_difference_proto.h)构图方式调用AdjacentDifference算子。 |
