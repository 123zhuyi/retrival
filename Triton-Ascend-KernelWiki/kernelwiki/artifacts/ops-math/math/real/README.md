# Real

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：返回输入tensor每个元素的实部。
- 计算公式：

  $$
  output_i = \mathrm{Re}(input_i)
  $$

  其中：
  - 当 $input_i = a_i + b_i \cdot j$ 为复数时，$output_i = a_i$；
  - 当 $input_i$ 为实数时，$output_i = input_i$（透传）。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 280px">
<col style="width: 360px">
<col style="width: 100px">
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
    <td>input</td>
    <td>输入</td>
    <td>待取实部的输入张量，公式中的 $input_i$。</td>
    <td>FLOAT、FLOAT16、COMPLEX64、COMPLEX32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>Tout</td>
    <td>可选属性</td>
    <td><ul><li>期望的输出dtype枚举值。</li><li>默认值为DT_FLOAT；实际输出类型由input dtype推导。</li></ul></td>
    <td>INT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>output</td>
    <td>输出</td>
    <td>取实部后的输出张量，公式中的 $output_i$，shape与input一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

input / output dtype对应关系：

| input | output |
| :--- | :--- |
| FLOAT | FLOAT |
| FLOAT16 | FLOAT16 |
| COMPLEX64 | FLOAT |
| COMPLEX32 | FLOAT16 |

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_real](./examples/test_aclnn_real.cpp) | 通过[aclnnReal](./docs/aclnnReal.md)接口方式调用Real算子。 |
| 图模式调用 | [test_geir_real](./examples/test_geir_real.cpp) | 通过[算子IR](./op_graph/real_proto.h)构图方式调用Real算子。 |
