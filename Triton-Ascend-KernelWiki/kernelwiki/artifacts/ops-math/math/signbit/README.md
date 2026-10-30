# Signbit

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：判断输入张量中每个元素的符号位是否为1，即元素是否为负数或-0.0。
- 计算公式：

  $$
  y_i = (x_i < 0) \: ? \: True : False \quad (当x_i为-0.0时，返回True)
  $$

- 示例：
  - 若x=[0, -3, 5]，signbit(x)的结果是[False, True, False]。
  - 若x=[[-0.0, 3.14], [-2.718, inf]]，signbit(x)的结果是[[True, False], [True, False]]。

## 参数说明

<table style="undefined;table-layout: fixed; width: 920px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
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
      <td>待进行符号位判断的入参，公式中的x_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、UINT8、INT32、INT64、UINT64、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>符号位判断的出参，公式中的y_i。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_signbit](./examples/test_aclnn_signbit.cpp) | 通过[aclnnSignbit](./docs/aclnnSignbit.md)接口方式调用Signbit算子。 |
