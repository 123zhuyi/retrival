# Spence

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：计算Spence函数（dilogarithm），S(x) = -integral_0^x ln(1-t)/t dt。

- 计算公式：

$$\text{Spence}(x) = \int_1^x \frac{\ln t}{t - 1} \, dt = \text{Li}_2(1 - x)$$

特殊值：

- $x < 0$ 时输出 NaN
- $x = 0$ 时输出 $\pi^2/6 \approx 1.6449$
- $x = 1$ 时输出 $0$

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
      <td>Spence函数的自变量，对应公式中x。x &lt; 0的元素输出NaN。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Spence函数的计算结果，对应公式中Spence(x)。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- FLOAT16、BFLOAT16输入在kernel内部提升到FLOAT32计算，结果再转回原始数据类型。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_spence](./examples/test_geir_spence.cpp) | 通过图模式调用Spence算子。 |
