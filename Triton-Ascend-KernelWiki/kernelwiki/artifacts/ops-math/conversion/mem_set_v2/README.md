# MemSetV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Ascend 950PR/Ascend 950DT</term>     |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|×|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|×|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|

## 功能说明

- 算子功能：给下游算子指定的output和workspace初始化成指定值。

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
      <td>是框架传递的待初始化的Tensor。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
      BF16、FLOAT16、FLOAT、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values_int</td>
      <td>属性</td>
      <td>指定对应位置的tensor的int类型的初始值。</td>
      <td>int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>values_float</td>
      <td>属性</td>
      <td>指定对应位置的tensor的float类型的初始值。</td>
      <td>float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>是框架传递的待初始化的Tensor，本算子的输出就是输入，原地进行初始化</td>
      <td>输入Tensor相同x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 无。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_mem_set_v2](./examples/test_geir_mem_set_v2.cpp)   | 通过[算子IR](./op_graph/mem_set_v2_proto.h)构图方式调用MemSetV2算子。 |
