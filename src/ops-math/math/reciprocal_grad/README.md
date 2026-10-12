# ReciprocalGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √    |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算 Reciprocal 算子的梯度。其中 `y = Reciprocal(x) = 1/x`，`dy` 为上游梯度，`z` 为对输入 `x` 的梯度。
- 计算公式：

$$
z = -y \times y \times dy
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
      <td>y</td>
      <td>输入</td>
      <td>公式中的y，前向 Reciprocal 算子的输出。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的dy，上游梯度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>公式中的z，对输入x的梯度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：不支持BFLOAT16。

## 约束说明

- y、dy、z 三者的数据类型必须一致。
- y、dy、z 三者的 shape 必须一致。
- 数据格式仅支持 ND。
- shape 维度不大于 8 维。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_reciprocal_grad](./examples/test_geir_reciprocal_grad.cpp) | 通过[算子IR](./op_graph/reciprocal_grad_proto.h)构图方式调用ReciprocalGrad算子。 |
