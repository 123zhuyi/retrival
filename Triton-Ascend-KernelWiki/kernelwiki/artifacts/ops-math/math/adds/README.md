# Adds

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：实现张量与标量的加法操作，对输入张量的每个元素加上一个标量值，输出结果张量。
- 计算公式：

  $$
  y_i = x_i + scalar
  $$

  其中：
  - $x_i$：输入张量的第i个元素
  - $scalar$：标量值
  - $y_i$：输出张量的第i个元素

- 使用场景：深度学习模型中的标量加法操作，常见于偏置添加、常数调整等场景。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>输入张量，对应公式中的x。支持1-8维Tensor。</td>
    <td>FLOAT16、FLOAT32、BFLOAT16、INT16、INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scalar</td>
    <td>输入</td>
    <td>标量值，对应公式中的scalar。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出张量，对应公式中的y。数据类型与输入张量x保持一致，shape与x一致。</td>
    <td>FLOAT16、FLOAT32、BFLOAT16、INT16、INT32、INT64</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- **数据类型约束**：输入张量x和输出张量y的数据类型必须一致。
- **shape约束**：输出张量y的shape必须与输入张量x的shape完全一致。
- **维度约束**：输入输出张量支持1-8维。

## 调用说明

| 调用方式 | 样例代码  | 说明     |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test\_geir\_adds](./examples/test_geir_adds.cpp) | 通过[算子IR](../../common/inc/op_graph/op_math_proto_extend.h)构图方式调用Adds算子。 |
