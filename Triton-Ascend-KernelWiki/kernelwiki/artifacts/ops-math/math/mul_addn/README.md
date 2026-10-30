# MulAddn

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |    √     |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |  
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：

  实现n>=2个mul和addn融合计算，减少搬运时间和内存的占用。

- 计算公式：
  
  输入x1，  x2为变长输入，为N个tensor组成的列表。x1输入中的每个tensor shape都为[B, M, 1], x2输入中的每个tensor shape都为[B, 1, K], y为输出，shape为[B, M, K]; N对应为addn算子的n数量，也为融合算子融合mul的数量。

$$
x1 * x2 = y
$$

$$
x1 = [[B, M, 1], [B, M, 1],...,[B, M, 1]] (共N个[B, M, 1])
$$

$$
x2 = [[B, 1, K],[B, 1, K],...,[B, 1, K]] (共N个[B, 1, K])
$$

$$
[[B, M, 1], [B, M, 1],...,[B, M, 1]] * [[B, 1, K],[B, 1, K],...,[B, 1, K]] = y
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 855px"><colgroup>
  <col style="width: 164px">
  <col style="width: 138px">
  <col style="width: 300px">
  <col style="width: 133px">
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
      <td>公式中的输入x1。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>N</td>
      <td>属性</td>
      <td>融合算子mul的数量。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

- K<=2040;
- 输入x1和x2的shape分别为[B,M,1], [B,1,K];
- N>=2;

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_mul_addn](./examples/test_geir_mul_addn.cpp)   | 通过[算子IR](./op_graph/mul_addn_proto.h)构图方式调用MulAddn算子。 |
