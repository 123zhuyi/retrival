# Neg

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入的每个元素完成相反数计算。

- 计算公式：
  $$
    output_i = (-1) * self_i
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>公式中的输出张量self。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT32、INT8、INT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>取反输出，公式中的输出张量output。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT32、INT8、INT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_neg](./examples/test_aclnn_neg.cpp) | 通过[aclnnNeg](./docs/aclnnNeg&aclnnInplaceNeg.md)接口方式调用Neg算子。    |
| 图模式调用 | [test_geir_neg](./examples/test_geir_neg.cpp) | 通过[算子IR](./op_graph/neg_proto.h)构图方式调用Neg算子。 |
