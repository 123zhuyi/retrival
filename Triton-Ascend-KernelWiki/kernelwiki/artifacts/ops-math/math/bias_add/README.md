# BiasAdd

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>           |  ×   |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：为输入张量的每一个元素加上一个偏差值。

- 计算公式：

$$out_i= x_i + bias_i$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 330px">
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
      <td>待进行BiasAdd计算的入参，公式中的x_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64</td>
      <td>NCHW、NHWC、NDHWC、NCDHW、ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>累加偏差，公式中的bias_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行BiasAdd计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64</td>
      <td>NCHW、NHWC、NDHWC、NCDHW、ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品：不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_bias_add](./examples/test_geir_bias_add.cpp)   | 通过[算子IR](./op_graph/bias_add_proto.h)构图方式调用BiasAdd算子。 |
