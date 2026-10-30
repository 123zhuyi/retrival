# CircularPadGrad

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

- 算子功能：CircularPad的反向传播操作。
- CircularPad2dBackward：计算CircularPad2d的反向传播。
- CircularPad3dBackward：计算CircularPad3d的反向传播。

## 参数说明

### CircularPad2dBackward

<table style="undefined;table-layout: fixed; width: 1007px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 284px">
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
    <td>gradOutput</td>
    <td>输入张量</td>
    <td>反向时输入的梯度数据，shape需要与circular_pad2d正向传播的output一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入张量</td>
    <td>正向时待填充的原输入数据，shape与gradInput一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>输入数组</td>
    <td>正向时填充的维度，长度为4，数值依次代表左右上下需要填充的值。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>输出张量</td>
    <td>反向时输出的梯度数据，shape与self一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

### CircularPad3dBackward

<table style="undefined;table-layout: fixed; width: 1007px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 284px">
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
    <td>gradOutput</td>
    <td>输入张量</td>
    <td>反向时输入的梯度数据，shape需要与circular_pad3d正向传播的output一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入张量</td>
    <td>正向时待填充的原输入数据，shape与gradInput一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>输入数组</td>
    <td>正向时填充的维度，长度为6，数值依次代表左右上下前后需要填充的值。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>输出张量</td>
    <td>反向时输出的梯度数据，shape与self一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- gradOutput的最后一维在不同类型下的大小需满足如下约束：
  - float16/bfloat16：(0, 16384)
  - float32：(0, 24576)
- padding值必须小于对应维度的大小。
- 输入和输出的数据类型必须一致。
- gradOutput的shape必须与正向传播的output一致。
- gradInput的shape必须与self一致。

## 调用说明

| 调用方式  | 样例代码                                                  | 说明                                                         |
| --------- |-------------------------------------------------------| ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_circular_pad_grad](./examples/test_aclnn_circular_pad_grad.cpp)       | 通过[aclnnCircularPad2dBackward](docs/aclnnCircularPad2dBackward.md)接口方式调用CircularPad2dBackward算子。 |
| aclnn接口 | [test_aclnn_circular_pad3d_backward](./examples/test_aclnn_circular_pad3d_backward.cpp) | 通过[aclnnCircularPad3dBackward](docs/aclnnCircularPad3dBackward.md)接口方式调用CircularPad3dBackward算子。 |
