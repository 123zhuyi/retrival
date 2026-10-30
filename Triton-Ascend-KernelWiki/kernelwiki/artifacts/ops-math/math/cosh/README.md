# Cosh

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |      √   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×    |
| <term>Atlas 推理系列产品</term>                       |     √    |
| <term>Atlas 训练系列产品</term>                       |     √    |

## 功能说明

- 接口功能：对输入矩阵的每个元素进行双曲余弦操作后输出。

- 计算公式：

$$
out_{i}=cosh(input_{i})
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
      <td>待进行cosh计算的入参，公式中的input_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行cosh计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 说明                                                           |
|--------------|--------------------------------------------------------------|
| aclnn调用 | 通过[aclnnCosh](./docs/aclnnCosh&aclnnInplaceCosh.md)接口方式调用Cosh算子。 |
| 图模式调用 | 通过[算子IR](op_graph/cosh_proto.h)构图方式调用Cosh算子。             |
