# AcoshGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √      |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |   √     |

## 功能说明

- 算子功能：计算Acosh（反双曲余弦）算子的反向梯度。
- 算子公式：

  $$
  z_i = dy_i \cdot \dfrac{1}{\sqrt{y_i^2 - 1}}
  $$

  其中：
  - $y_i$为前向Acosh算子的输入张量，值域期望$\geq 1$；
  - $dy_i$为上游传入的梯度；
  - $z_i$为对原始输入张量的梯度，等于上游梯度乘以$1/\sqrt{y_i^2 - 1}$。

- 边界（$y_i \leq 1$）：$y_i = 1$时$\sqrt{0}=0$，除零结果为$+\text{Inf}$；$y_i < 1$时$\sqrt{负数}$结果为NaN，按IEEE 754自然传播。建议输入$y$的值域落在$[1, +\infty)$，以避免产生Inf或NaN结果。

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
      <td>y</td>
      <td>输入</td>
      <td>前向Acosh算子的输入张量。值域期望落在[1, +∞)。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>上游传入的梯度张量，shape与dtype与y一致。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>对原始输入张量的梯度，shape与dtype与y一致。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- y与dy的shape必须完全一致。
- y与dy的dtype必须完全一致。
- 仅支持ND格式。
- 支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，非连续的Tensor维度不大于8。

## 调用说明

| 调用方式 | 调用样例                                                                      | 说明                                                                                       |
| -------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 图模式调用 | [test_geir_acosh_grad](./examples/test_geir_acosh_grad.cpp)             | 通过[算子IR](./op_graph/acosh_grad_proto.h)构图方式调用AcoshGrad算子。                 |
