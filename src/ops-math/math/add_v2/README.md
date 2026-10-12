# AddV2

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    √     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：对两个输入张量`x1`和`x2`执行逐元素加法，兼容TensorFlow AddV2语义，支持广播（broadcast）。

- 计算公式：

$$y_i = x1_i + x2_i$$

- 广播示例：

```text
x1 shape (3, 4), x2 shape (1, 4) -> y shape (3, 4)
x1 shape (3, 1), x2 shape (1, 4) -> y shape (3, 4)
```

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>x1</td>
      <td>输入</td>
      <td>加法运算的第一个输入张量，公式中的x1_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT16、UINT8、INT8、INT64、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>加法运算的第二个输入张量，公式中的x2_i。数据类型需与x1一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT16、UINT8、INT8、INT64、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>加法运算的输出张量，公式中的y_i。数据类型与x1一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT16、UINT8、INT8、INT64、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x1和x2的shape需满足广播规则。
- 输入x1和x2的数据类型需相同，不支持混合数据类型输入。
- 输入数据类型需在支持列表内，不支持DOUBLE、COMPLEX128、BOOL、COMPLEX32。
- 支持空Tensor。当输出y的元素个数为0时，算子不下发有效计算，直接返回成功。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_add_v2](./examples/test_geir_add_v2.cpp)   | 通过[算子IR](./op_graph/add_v2_proto.h)构图方式调用AddV2算子。 |
