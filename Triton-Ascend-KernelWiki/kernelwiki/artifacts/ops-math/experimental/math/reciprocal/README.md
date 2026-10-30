# Reciprocal

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：完成倒数的计算。

- 计算公式：

$$
out = {\frac{1} {input}}
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
      <td>待进行倒数计算的入参，公式中的input。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行完倒数计算的出参，公式中的out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>  
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_reciprocal.cpp](./examples/test_aclnn_reciprocal.cpp) | 通过[aclnnReciprocal&aclnnInplaceReciprocal](./docs/aclnnReciprocal&aclnnInplaceReciprocal.md)接口方式调用reciprocal算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| fulltower | 个人开发者 | Reciprocal | 2026/2/27 | reciprocal算子适配开源仓 |
| 中国海洋大学王胜科团队 | 个人开发者 | reciprocal | 2025/7/30 | 新增Reciprocal算子 |
