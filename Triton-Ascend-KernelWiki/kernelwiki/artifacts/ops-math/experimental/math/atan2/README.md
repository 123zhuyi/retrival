# Atan2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：计算两个输入张量 x1（分子，对应数学中的 y）和 x2（分母，对应数学中的 x）的反正切值，返回值的角度范围为 (−π, π]，可正确区分所有象限。

- 计算公式：

$$
\text{out}_i = \text{atan2}(x1_i,\ x2_i)
$$

其中 $\text{atan2}(y, x)$ 定义为：

$$
\text{atan2}(y, x) = \begin{cases}
\arctan(y/x) & x > 0 \\
\arctan(y/x) + \pi & x < 0,\ y \ge 0 \\
\arctan(y/x) - \pi & x < 0,\ y < 0 \\
+\pi/2 & x = 0,\ y > 0 \\
-\pi/2 & x = 0,\ y < 0 \\
0 & x = 0,\ y = 0
\end{cases}
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
      <td>x1</td>
      <td>输入</td>
      <td>待计算的分子张量，对应数学公式中的 y。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待计算的分母张量，对应数学公式中的 x。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算结果张量，与输入 shape 相同，值域为 (−π, π]。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x1 与 x2 的 shape 必须一致。
- x1 与 x2 的数据类型必须一致。
- 输出 y 的 shape 与 x1 相同。
- 典型场景建议尾轴为 32B 对齐（float32 为 8 的倍数，float16 为 16 的倍数）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| aclnn调用 | [test_aclnn_atan2.cpp](./examples/test_aclnn_atan2.cpp) | 通过[aclnnAtan2](./docs/aclnnAtan2.md)接口方式调用Atan2算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Tream | 个人开发者 | Atan2 | 2025/03 | Atan2 算子适配开源仓 |
