# Tan

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：逐元素计算输入张量的正切值，输出张量的shape与输入张量相同。

- 计算公式：

  $$
  y_i = \tan(x_i)
  $$

  其中，$x_i$表示输入张量`x`的第$i$个元素，$y_i$表示输出张量`y`的第$i$个元素。

## 参数说明

<table style="table-layout: fixed; width: 910px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入张量x。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y，shape与x相同。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：AI Core支持FLOAT16、FLOAT、BFLOAT16、INT32。
- <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：AI Core支持FLOAT16、FLOAT、INT32，不支持BFLOAT16。
- DOUBLE、COMPLEX64、COMPLEX128通过AICPU路径计算。

## 约束说明

- 输入与输出的shape必须一致，数据格式为ND。
- 在<term>Ascend 950PR/Ascend 950DT</term>的AI Core实现中，输入为NaN、Inf或绝对值大于等于$10^7$时，输出为NaN。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| aclnn API | [test_aclnn_tan.cpp](./examples/test_aclnn_tan.cpp) | 通过[aclnnTan和aclnnInplaceTan](./docs/aclnnTan&aclnnInplaceTan.md)接口调用Tan算子。 |
| GE图模式 | [test_geir_tan.cpp](./examples/test_geir_tan.cpp) | 通过[算子IR](./op_graph/tan_proto.h)构图调用Tan算子。 |
