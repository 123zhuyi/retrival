# PadV3GradReplication

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

- 算子功能：replication_pad3d的反向传播。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 922px"><colgroup>
<col style="width: 168.333333px">
<col style="width: 116.666666px">
<col style="width: 305.666666px">
<col style="width: 233.666666px">
<col style="width: 97.666666px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0lax">描述</th>
    <th class="tg-0lax">数据类型</th>
    <th class="tg-0lax">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">gradOutput</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0lax">replication_pad3d的正向传播，shape支持5维且维度需要与self和gradInput保持一致。</td>
    <td class="tg-0lax">FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">self</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0lax">shape支持5维且维度需要与self和gradInput保持一致，shape与gradInput一致。</td>
    <td class="tg-0lax">FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">padding</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0lax">padding长度为6，表示三维张量左右上下前后需要填充的值，padding的前两个值都需要小于self最后一维度的数值，第三和第四个值小于self倒数第二维度的数值，第五和第六两个值小于self倒数第三维度的数值。</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0pky">gradInput</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0lax">数据类型与self保持一致。</td>
    <td class="tg-0lax">FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
    <td class="tg-0lax">ND</td>
  </tr>
</tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_pad_v3_grad_replication](examples/test_aclnn_pad_v3_grad_replication.cpp) | 通过[aclnnReplicationPad3dBackward](docs/aclnnReplicationPad3dBackward.md)接口方式调用PadV3GradReplication算子。    |
