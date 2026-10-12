# Bias

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

Bias 按照 `axis`、`num_axes` 和 `bias_from_blob` 推导 `bias` 的逻辑广播形状，并计算 `y = x + broadcast(bias)`。

计算公式如下：

$$y_i = x_i + bias_i$$

`broadcast(bias)` 的推导规则如下：

- `axis < 0` 时，先执行 `axis += rank(x)`。
- `bias_from_blob = true` 且 `num_axes = -1` 时，`broadcast_shape = [1] * axis + shape(bias)`。
- `bias_from_blob = true` 且 `num_axes = 0` 时，`broadcast_shape = [1] * rank(x)`。
- `bias_from_blob = true` 且 `num_axes > 0` 时，`broadcast_shape = [1] * axis + shape(bias) + [1] * (rank(x) - axis - num_axes)`。
- `bias_from_blob = false` 且 `bias` 为单元素时，`broadcast_shape = [1] * rank(x)`。
- `bias_from_blob = false` 的其它场景，`broadcast_shape = [1] * axis + shape(bias) + [1] * (rank(x) - axis - rank(bias))`。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 380px">
  <col style="width: 250px">
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
      <td>待进行Bias计算的输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NC1HWC0</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>按属性广播后与x相加的偏置张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NC1HWC0</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>bias 逻辑形状的起始对齐轴，默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_axes</td>
      <td>属性</td>
      <td>bias 覆盖的轴数，默认值为1；支持-1、0和正整数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias_from_blob</td>
      <td>属性</td>
      <td>是否按 blob bias 规则推导逻辑广播形状，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，形状和数据类型跟随x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NC1HWC0</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持 ND 和 NC1HWC0 数据格式（4D输入时支持NC1HWC0），以及 FLOAT、FLOAT16、BFLOAT16 同数据类型输入输出；不支持混合数据类型。
- `bias` 经过属性推导后的每个维度必须为1，或与`x`的对应维度相同。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_bias](./examples/test_geir_bias.cpp) | 通过[算子IR](./op_graph/bias_proto.h)构图方式调用Bias算子。 |
