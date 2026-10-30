# BitwiseAnd

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：对输入x1和x2做与运算。

- 计算公式：

$$
y = x1 \; and \; x2
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
      <td>待进行BitwiseAnd计算的入参，公式中的x1。</td>
      <td>int16,uint16,int32</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行BitwiseAnd计算的入参，公式中的x2。</td>
      <td>int16,uint16,int32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行BitwiseAnd计算的出参，公式中的输出。</td>
      <td>int16,uint16,int32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_bitwise_and_tensor](./examples/test_aclnn_bitwise_and_tensor.cpp) | 通过[aclnnBitwiseAndTensor](./docs/aclnnBitwiseAndTensor.md)接口方式调用SwishGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| ilovescrapy | 个人开发者 | BitwiseAnd | 2025/12/26 | BitwiseAnd算子适配开源仓 |
