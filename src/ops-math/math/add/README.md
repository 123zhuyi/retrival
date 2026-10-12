# Add

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR/Ascend 950DT</term>       |   √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |   √  |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |   √  |
| <term>Atlas 200I/500 A2 推理产品</term>          |   ×  |
| <term>Atlas 推理系列产品</term>                    |   ×  |
| <term>Atlas 训练系列产品</term>                    |   √  |

## 功能说明

- 算子功能：对输入完成相加操作。
- 计算公式：
  $$
  y = x_1 + x_2
  $$

## 参数说明

<table style="table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 320px">
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
      <td>公式中的输入张量x_1</td>
      <td>BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT64, FLOAT16, BFLOAT16, FLOAT32, COMPLEX128, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入张量x2</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持 STRING 数据类型。算子IR（add\_proto.h）中注册 DT\_STRING 仅为兼容 TensorFlow Add 算子，图模式（GEIR）调用通路虽可通过类型推断，但无对应kernel实现，运行时会失败。
- COMPLEX32 仅 Ascend 950 内核实现支持，aclnnAdd / aclnnInplaceAdd 接口及图模式通路均不支持，请使用 COMPLEX64 / COMPLEX128。

## 调用说明

| 调用方式    | 调用样例                                              | 说明                                                                           |
| ------- | ------------------------------------------------- | ---------------------------------------------------------------------------- |
| aclnn调用 | [test\_aclnn\_add](./examples/test_aclnn_add.cpp) | 通过[aclnnAdd和aclnnInplaceAdd](./docs/aclnnAdd\&aclnnInplaceAdd.md)接口方式调用Add算子 |
| 图模式调用   | [test\_geir\_add](./examples/test_geir_add.cpp)   | 通过[算子IR](./op_graph/add_proto.h)构图方式调用Add算子                                  |
