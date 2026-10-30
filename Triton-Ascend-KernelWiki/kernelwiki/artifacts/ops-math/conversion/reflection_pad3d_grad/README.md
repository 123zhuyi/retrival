# ReflectionPad3dGrad

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

- 算子功能：计算aclnnReflectionPad3d api的反向传播。

## 参数说明

<table style="undefined;table-layout: fixed; width: 866px"><colgroup>
<col style="width: 139px">
<col style="width: 148px">
<col style="width: 236px">
<col style="width: 267px">
<col style="width: 76px">
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
    <td>输入</td>
    <td>反向传播的输入。</td>
    <td>FLOAT16、FLOAT32、DOUBLE、 COMPLEX64、COMPLEX128</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入</td>
    <td>正向的输入张量。</td>
    <td>FLOAT16、FLOAT32、DOUBLE、 COMPLEX64、COMPLEX128</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>输入</td>
    <td>长度为6，数值依次代表左右上下前后需要填充的值。</td>
    <td>aclIntArray数组</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>输出</td>
    <td>反向传播的输出。</td>
    <td>FLOAT16、FLOAT32、DOUBLE、 COMPLEX64、COMPLEX128</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- gradOutput、self和gradInput的维度需一致（支持四维或五维），且它们的形状需与reflection_pad3d正向传播的输出形状相互一致。
- 输入值域限制：padding前两个数值需小于self最后一维度的数值，中间两个数值需小于self倒数第二维度的数值，后两个数值需小于self倒数第三维度的数值。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_reflection_pad3d_grad](./examples/test_aclnn_reflection_pad3d_grad.cpp) | 通过[aclnnReflectionPad3dBackward](docs/aclnnReflectionPad3dBackward.md)接口方式调用ReflectionPad3dGrad算子。 |
