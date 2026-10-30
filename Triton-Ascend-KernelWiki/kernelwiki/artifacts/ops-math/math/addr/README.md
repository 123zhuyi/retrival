# Addr

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：求一维向量vec1和vec2的外积得到一个二维矩阵，并将外积结果矩阵乘一个系数后和自身乘系数相加后输出

- 计算公式：
  $$
  \text{out} = \beta\ \text{self} + \alpha\ (\text{vec1} \otimes\text{vec2})
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
      <td>待进行addr计算的入参，公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行addr计算的入参，公式中的vec1。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>待进行addr计算的入参，公式中的vec2。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>待进行addr计算的入参，公式中的β。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>待进行addr计算的入参，公式中的α。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行addr计算的出参，公式中的out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品、Ascend 950PR/Ascend 950DT：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_addr](./examples/test_aclnn_addr.cpp) | 通过[aclnnAddr](docs/aclnnAddr&aclnnInplaceAddr.md)接口方式调用Addr算子。 |
