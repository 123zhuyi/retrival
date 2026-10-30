# Dot

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

- 接口功能：计算两个一维张量的点积结果。

- 计算公式：

$$
self = [x_{1}, x_{2}, ..., x_{n}]
$$

$$
tensor = [y_{1}, y_{2}, ..., y_{n}]
$$

$$
out = x_{1}*y_{1} + x_{2}*y_{2} + ... + x_{n}*y_{n}
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
      <td>self</td>
      <td>输入</td>
      <td>参与点积计算的输入张量，公式中的self。与tensor、out的数据类型一致，与tensor的shape一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>tensor</td>
      <td>输入</td>
      <td>参与点积计算的输入张量，公式中的tensor。与self、out的数据类型一致，与self的shape一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>点积计算结果，公式中的输出out。与self、tensor的数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dot](./examples/test_aclnn_dot.cpp) | 通过[aclnnAbs](./docs/aclnnDot.md)接口方式调用Dot算子。 |
