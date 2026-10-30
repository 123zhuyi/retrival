# Square

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：为输入张量的每一个元素计算平方值。

- 计算公式：

  $$
  y = x * x = x^2
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>公式中的输入张量x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：不支持BFLOAT16。

## 调用说明

| 调用方式 | 调用样例                                                  | 说明                                                                 |
|---------|-------------------------------------------------------|--------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_square](./examples/arch35/test_aclnn_square.cpp) | 通过[aclnnSquare](./docs/aclnnSquare.md)接口方式调用Square算子。 |
| 图模式调用 | [test_geir_square](./examples/test_geir_square.cpp)     | 通过[算子IR](./op_graph/square_proto.h)构图方式调用Square算子。                   |
