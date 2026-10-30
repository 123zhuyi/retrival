# Ndtri

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

- 算子功能：逆标准正态累积分布函数（probit / inverse normal CDF）。对输入概率张量逐元素计算标准正态分位点，底层采用Cephes数学库的分区间有理逼近算法。

- 计算公式：

$$y_i = \mathrm{ndtri}(x_i) = \Phi^{-1}(x_i) = \sqrt{2} \cdot \mathrm{erf}^{-1}(2 x_i - 1), \quad x_i \in (0, 1)$$

- 特殊值处理：

| 输入 $x_i$ | 输出 $y_i$ |
| :---: | :---: |
| `0` | `-inf` |
| `1` | `+inf` |
| $x_i < 0$ 或 $x_i > 1$ | `NaN` |
| `NaN` / `+inf` / `-inf` | `NaN` |

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
      <td>待进行ndtri计算的概率张量，公式中的x_i，推荐值域(0, 1)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>逆标准正态CDF的计算结果，公式中的y_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                       | 说明                                                              |
|--------------|--------------------------------------------|-----------------------------------------------------------------|
| 图模式调用 | [test_geir_ndtri](./examples/test_geir_ndtri.cpp) | 通过[算子IR](./op_graph/ndtri_proto.h)构图方式调用Ndtri算子。 |
