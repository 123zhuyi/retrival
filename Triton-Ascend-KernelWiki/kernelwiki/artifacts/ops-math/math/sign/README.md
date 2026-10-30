# Sign

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |  ×     |

## 功能说明

- 算子功能：对输入的tensor逐元素进行Sign符号函数的运算并输出结果tensor。

- 计算公式：

  $$
  resultput_i = \left\{
  \begin{aligned}
  1,\quad input_i > 0\\
  0,\quad input_i = 0\\
  -1,\quad input_i < 0
  \end{aligned}
  \right.
  $$

- 计算公式（BOOL类型情况）：

  $$
  resultput_i = \left\{
  \begin{aligned}
  \text{True},\quad input_i = \text{True}\\
  \text{False},\quad input_i = \text{False}\\
  \end{aligned}
  \right.
  $$

- 计算公式（复数情况，其中real和下面各分别表示取实部和虚部）：

  $$
  resultput_i = \alpha \cdot cos(\theta_i) + \beta \cdot sin(\theta_i) \\
  \alpha = \left\{
  \begin{aligned}
  1,\quad real(input_i) > 0 \\
  0,\quad real(input_i) = 0 \\
  -1,\quad real(input_i) < 0 \\
  \end{aligned}
  \right.
  \\
  \beta = \left\{
  \begin{aligned}
  1,\quad image(input_i) > 0 \\
  0,\quad image(input_i) = 0 \\
  -1,\quad image(input_i) < 0 \\
  \end{aligned}
  \right.
  \\
  \theta_i = arctan(\frac{image(input_i)}{real(input_i)})
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
      <td>待进行sign计算的入参，公式中的input_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行sign计算的出参，公式中的resultput_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sign](./examples/test_aclnn_sign.cpp) | 通过[aclnnSign](./docs/aclnnSign.md)接口方式调用Sign算子。 |
