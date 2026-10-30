# StridedSliceAssignV2

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：StridedSliceAssignV2是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。
  inputValue的shape第i维的计算公式为：$inputValueShape[i] = \lceil\frac{end[i] - begin[i]}{strides[i]} \rceil$，其中$\lceil x\rceil$ 表示对 $x$向上取整。$end$ 和 $begin$ 为经过特殊值调整后的取值，调整方式为：当 $end[i] < 0$ 时，$end[i]=varShape[i] + end[i]$ ，若仍有$end[i] < 0$，则 $end[i] = 0$ ，当 $end[i] > varShape[i]$ 时， $end[i] = varShape[i]$ 。$begin$ 同理。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1162px"><colgroup>
<col style="width: 139px">
<col style="width: 148px">
<col style="width: 532px">
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
    <td>varRef</td>
    <td>输入|输出张量</td>
    <td>输入的tensor。</td>
    <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>inputValue</td>
    <td>输入张量</td>
    <td>替换切片的tensor，数据类型需与varRef保持一致，shape需要与varRef计算得出的切片shape保持一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>begin</td>
    <td>输入数组</td>
    <td>切片位置的起始索引。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>end</td>
    <td>输入数组</td>
    <td>切片位置的终止索引。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>输入数组</td>
    <td>切片的步长。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>axesOptional</td>
    <td>输入数组</td>
    <td>可选参数，切片的轴。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
</tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

无。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_strided_slice_assign_v2](./examples/test_aclnn_strided_slice_assign_v2.cpp) | 通过[aclnnStridedSliceAssignV2](docs/aclnnStridedSliceAssignV2.md)接口方式调用StridedSliceAssignV2算子。 |
