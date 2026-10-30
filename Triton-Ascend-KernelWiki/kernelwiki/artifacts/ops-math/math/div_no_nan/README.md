# DivNoNan

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √    |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √    |

## 功能说明

- 算子功能：完成安全除法计算,在处理除法时避免分母为0导致的NaN值问题。当分母为0时,返回0而不是NaN。
- 计算公式：
  $$
  y = \begin{cases}
  \frac{x1}{x2}, & \text{if } x2 \neq 0 \\
  0, & \text{if } x2 = 0
  \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
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
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1,作为分子。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、UINT8、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2,作为分母。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、UINT8、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、UINT8、INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式 | [test_geir_div_no_nan](examples/test_geir_div_no_nan.cpp) | 通过[算子IR](op_graph/div_no_nan_proto.h)构图方式调用DivNoNan算子。 |
