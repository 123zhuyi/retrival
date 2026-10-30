# DiagPart

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：提取输入矩阵的对角线元素。

- 计算公式：

$$
y_i = x_{i,i}
$$

对于输入形状为 [N, N] 的矩阵 x，输出形状为 [N] 的向量 y，其中 y[i] = x[i][i]。

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
      <td>待提取对角线元素的输入矩阵，形状必须为 [N, N] 的方阵。</td>
      <td>float16,float32,int32</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>提取的对角线元素，形状为 [N] 的向量。</td>
      <td>float16,float32,int32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量 x 必须是二维方阵，形状为 [N, N]。
- 输入和输出的数据类型必须一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_diag_part](./examples/test_aclnn_diag_part.cpp) | 通过[aclnnDiagPart](./docs/aclnnDiagPart.md)接口方式调用DiagPart算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| fangfangssj | 个人开发者 | DiagPart | 2026/4/15 | DiagPart算子适配开源仓 |
