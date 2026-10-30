# ComplexAbs

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

- 算子功能：计算输入复数张量中每一个元素的绝对值（模长）。

- 计算公式：

  对于复数输入 $x_i = a_i + b_i \cdot j$，输出其模长：

$$
y_{i}=|x_{i}|=\sqrt{a_{i}^{2}+b_{i}^{2}}
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
      <td>待进行复数绝对值计算的入参，公式中的输入张量x。</td>
      <td>COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>复数绝对值计算的出参，公式中的输出张量y。输出数据类型由输入数据类型决定：输入为COMPLEX64时输出FLOAT，输入为COMPLEX32时输出FLOAT16。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Tout</td>
      <td>属性</td>
      <td>表示输出的数据类型，兼容TensorFlow ComplexAbs算子保留，当前不使用，输出数据类型由输入数据类型推导得到。</td>
      <td>Type，默认值为DT_FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入只支持COMPLEX32、COMPLEX64类型，不支持其他数据类型。
- 输出数据类型由输入数据类型决定，不支持显式指定：COMPLEX64对应输出FLOAT，COMPLEX32对应输出FLOAT16。
- 输入与输出的shape一致，支持动态Shape（DynamicShape）与动态维度（DynamicRank）。
- 数据格式支持ND，支持非连续的Tensor，维度不大于8。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| 图模式调用 | [算子IR](./op_graph/complex_abs_proto.h) | 通过[算子IR](./op_graph/complex_abs_proto.h)构图方式调用ComplexAbs算子。 |
