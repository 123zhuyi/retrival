# Sqrt

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |   ×     |
| <term>Atlas 训练系列产品</term>                              |   √     |

## 功能说明

- 算子功能：完成非负数平方根计算，负数情况返回nan。
- 计算公式：

  $$
  out=sqrt(self)=\begin{cases}
  \sqrt {self}, & self\ge 0 , \\
  nan, &  self\lt 0
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
      <td>self</td>
      <td>输入</td>
      <td>待进行sqrt计算的入参，公式中的self。</td>
      <td>COMPLEX64、COMPLEX128、FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行sqrt计算的出参，公式中的out。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sqrt](./examples/test_aclnn_sqrt.cpp) | 通过[aclnnSqrt](docs/aclnnSqrt&aclnnInplaceSqrt.md)接口方式调用Sqrt算子。 |
| 图模式调用 | [test_geir_sqrt](./examples/test_geir_sqrt.cpp) | 通过[算子IR](./op_graph/sqrt_proto.h)构图方式调用Sqrt算子。 |
