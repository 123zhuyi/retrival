# Unpack

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √    |
| <term>Atlas 推理系列产品</term>                              |    √    |
| <term>Atlas 训练系列产品</term>                              |    √    |

## 功能说明

- 算子功能：将张量沿着某一维度拆分为多个子张量。

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
      <td>x</td>
      <td>输入</td>
      <td>输入tensor</td>
      <td>INT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BFLOAT16、INT16、UINT16、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num</td>
      <td>输入属性</td>
      <td>拆分的数量</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入属性</td>
      <td>拆分的维度</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输入张量在指定维度上的大小</td>
      <td>INT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BFLOAT16、INT16、UINT16、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_unpack](./examples/test_geir_unpack.cpp)   | 通过[算子IR](./op_graph/unpack_proto.h)构图方式调用Unpack算子。 |
