# SquareSumV1

## 产品支持情况

| 产品                                                                | 是否支持 |
|:------------------------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                            |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>                      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                               |    ×     |
| <term>Atlas 推理系列产品</term>                                         |    √     |
| <term>Atlas 训练系列产品</term>                                         |    √     |

## 功能说明

- 算子功能：用于计算输入张量在指定轴上的平方和。

- 计算公式：

$$
y = \sum_{i\in\text{axis}}x_i^2
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
      <td>公式中输入张量x。</td>
      <td>FLOAT16、FLOAT、BFloat16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中输出张量y。</td>
      <td>FLOAT16、FLOAT、BFloat16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>一个整数列表，指定在哪些轴上进行求和。</td>
      <td>INT32_ARRAY</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keep_dims</td>
      <td>属性</td>
      <td>是否在输出中保留减少的维度，默认为false。</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                | 说明                                                             |
|--------------|-----------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_square_sum_v1.cpp](examples/test_geir_square_sum_v1.cpp)   | 通过[算子IR](op_graph/square_sum_v1_proto.h)构图方式调用SquareSumV1算子。                   |
