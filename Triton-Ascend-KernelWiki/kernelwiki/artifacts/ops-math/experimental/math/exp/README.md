# Exp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：返回一个新的张量，该张量的每个元素都是输入张量对应元素的指数。
- 计算公式：

$$
out_{i} = e^{self_{i}}
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
      <td>待进行exp计算的入参，公式中的self_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>base</td>
      <td>可选属性</td>
      <td><ul><li>用于指定指数运算的底数。</li><li>默认值为-1.0。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选属性</td>
      <td><ul><li>用于指定缩放系数。</li><li>默认值为1.0。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>可选属性</td>
      <td><ul><li>用于指定偏移量。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行exp计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_exp](./examples/test_aclnn_exp.cpp) | 通过[aclnnExp&aclnnInplaceExp](./docs/aclnnExp&aclnnInplaceExp.md)接口方式调用Exp算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| fulltower | 个人开发者 | Exp | 2026/2/27 | Exp算子适配开源仓 |
| 中国海洋大学王胜科团队 | 个人开发者 | Exp | 2025/7/30 | 新增Exp算子 |
