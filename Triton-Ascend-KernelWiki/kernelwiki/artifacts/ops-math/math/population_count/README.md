# PopulationCount

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：逐元素统计输入张量中每个16比特整数的二进制表示中bit=1的个数（population count，亦称Hamming weight / popcount）。

- 计算公式：

$$y_i = \text{popcount}(x_i)$$

  其中：

- $x_i$：输入张量的第i个元素，按无符号16位宽解释（INT16负数按二进制补码逐比特计数）

- $y_i$：输出张量的第i个元素，取值范围为 $[0, 16]$

- 使用场景：汉明距离计算、稀疏性分析、位掩码处理等需要统计二进制位数的场景。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待统计的整数张量，对应公式中的x。支持0-8维Tensor，0维表示标量。</td>
    <td>INT16、UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>每个元素对应的位计数结果，对应公式中的y。shape与x一致。</td>
    <td>UINT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 输入x的数据类型必须为INT16或UINT16，输出y的数据类型固定为UINT8。
- 输出y的shape必须与输入x的shape完全一致，不支持广播。
- 支持0-8维Tensor，0维表示标量。
- 支持空Tensor（元素个数为0），直接返回空结果。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| 图模式调用 | [test\_geir\_population\_count](./examples/test_geir_population_count.cpp) | 通过[算子IR](../../common/inc/op_graph/op_math_proto_extend.h)构图方式调用PopulationCount算子。 |
