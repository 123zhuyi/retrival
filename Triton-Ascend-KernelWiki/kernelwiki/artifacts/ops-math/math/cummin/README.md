# Cummin

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：计算self中的累积最小值，并返回最小值以及对应的索引。

- 计算公式：

  valuesOut：

  $$
  valuesOut_{i} = min(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$

  indicesOut：

  $$
  indicesOut_{i} = argmin(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 290px">
  <col style="width: 450px">
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
      <td>公式中的self，输入Tensor。</td>
      <td>FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的valuesOut，x的累积最小值。</td>
      <td>FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmin</td>
      <td>输出</td>
      <td>公式中的indicesOut，y应的索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>处理维度</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn调用 | [test_aclnn_cummin](./examples/test_aclnn_cummin.cpp) | 通过[aclnnCummin](./docs/aclnnCummin.md)接口方式调用Cummin算子。    |
