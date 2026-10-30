# StridedSliceGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                              |    √    |
| <term>Atlas 训练系列产品</term>                              |    √    |

## 功能说明

- 算子功能：将子张量的梯度映射回原始张量的对应位置。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>shape</td>
      <td>输入</td>
      <td>输入tensor</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin</td>
      <td>输入</td>
      <td>起始位置</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>输入</td>
      <td>结束位置</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>输入</td>
      <td>步长</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>梯度</td>
      <td>FLOAT16、BFLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin_mask</td>
      <td>输入属性</td>
      <td>输入格式</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end_mask</td>
      <td>输入属性</td>
      <td>输入格式</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ellipsis_mask</td>
      <td>输入属性</td>
      <td>输入格式</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>new_axis_mask</td>
      <td>输入属性</td>
      <td>输入格式</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shrink_axis_mask</td>
      <td>输入属性</td>
      <td>输入格式</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>填充后的梯度张量</td>
      <td>FLOAT16、BFLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_strided_slice_grad](./examples/test_geir_strided_slice_grad.cpp)   | 通过[算子IR](./op_graph/strided_slice_grad_proto.h)构图方式调用StridedSliceGrad算子。 |
