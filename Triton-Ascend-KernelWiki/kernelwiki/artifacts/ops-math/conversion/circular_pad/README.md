# CircularPad

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：使用循环填充方式对输入张量进行填充操作。
- CircularPad2d：对输入张量的最后两维进行循环填充。
- CircularPad3d：对输入张量的最后三维进行循环填充。

## 参数说明

### CircularPad2d

<table style="undefined;table-layout: fixed; width: 974px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 251px">
<col style="width: 96px">
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
    <td>self</td>
    <td>输入张量</td>
    <td>待填充的原输入数据，shape支持3-4维。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>输入数组</td>
    <td>填充维度，长度为4，数值依次代表左右上下需要填充的值。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>填充后的输出结果，shape支持3-4维。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

### CircularPad3d

<table style="undefined;table-layout: fixed; width: 974px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 251px">
<col style="width: 96px">
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
    <td>self</td>
    <td>输入张量</td>
    <td>待填充的原输入数据，shape支持4-5维。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>输入数组</td>
    <td>填充维度，长度为6，数值依次代表左右上下前后需要填充的值。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>填充后的输出结果，shape支持4-5维。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

- padding值必须小于对应维度的大小。
- out的最后一维在不同类型下的大小需满足如下约束：
  - int8：(0, 98304)
  - float16/bfloat16：(0, 49152)
  - int32/float32：(0, 24576)
- 输入和输出的数据类型必须一致。
- 支持非连续的Tensor。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_circular_pad](./examples/test_aclnn_circular_pad.cpp) | 通过[aclnnCircularPad2d](docs/aclnnCircularPad2d.md), [aclnnCircularPad3d](docs/aclnnCircularPad3d.md)接口方式调用CircularPad2d算子。 |
