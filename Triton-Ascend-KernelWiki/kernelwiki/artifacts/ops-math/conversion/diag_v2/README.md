# DiagV2

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |
|Kirin X90 处理器系列产品|√|
|Kirin 9030 处理器系列产品|√|

## 功能说明

- 算子功能：根据输入的二维张量，提取由diagonal指定的对角线元素。
  
  如果diagonal = 0，选择主对角线；
  
  如果diagonal > 0，选择主对角线上方的第diagonal条对角线；
  
  如果diagonal < 0，选择主对角线下方的第-diagonal条对角线。

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
      <td>输入张量。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>可选属性</td>
      <td><ul><li>表示选择对角线的位置。</li><li>默认值为0，表示主对角线。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品：不支持BFLOAT16。
- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16、COMPLEX。

## 约束说明

- 输入必须为二维张量，输出的数据类型和输入一样。
- diagonal的取值应符合-(m-1) <= diagonal <= (n-1)，其中(m,n)为输入的shape。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_diag_v2](./examples/test_geir_diag_v2.cpp)   | 通过[算子IR](./op_graph/diag_v2_proto.h)构图方式调用DiagV2算子。 |
