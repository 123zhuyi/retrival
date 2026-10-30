# ConcatV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|√|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|√|

## 功能说明

- 算子功能：用于沿指定维度将多个输入Tensor进行拼接，输出包含所有输入数据按顺序拼接后的Tensor。
- 计算流程：
  - 输入：
    - Tensor列表x[0], x[1],…, x[N-1]
    - 拼接维度concat_dim

  - 流程：
    1. 校验所有输入Tensor数据类型一致；
    2. 校验除concat_dim外所有维度完全相同；
    3. 沿concat_dim维度依次拼接：
       y = ConcatV2(x[0], x[1], ..., x[N-1], axis = concat_dim)
  - 输出：拼接后的Tensor y

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>动态输入列表，流程图中的输入x[i]。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT32、UINT8、INT16、INT8、COMPLEX64、INT64、QINT8、QUINT8、QINT32、UINT16、COMPLEX128、UINT32、UINT64、QINT16、QUINT16、BOOL、STRING、HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concat_dim</td>
      <td>输入</td>
      <td>指定拼接维度，即计算流程中的concat_dim。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>N</td>
      <td>可选属性</td>
      <td>输入x的数量，默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算流程中的输出y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT32、UINT8、INT16、INT8、COMPLEX64、INT64、QINT8、QUINT8、QINT32、UINT16、COMPLEX128、UINT32、UINT64、QINT16、QUINT16、BOOL、STRING、HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品：不支持BFLOAT16。

## 约束说明

- 所有输入Tensor在除拼接维度外的形状必须一致。
- 输入列表"x"至少包含2 个Tensor。
- 拼接维度concat_dim必须在输入Tensor的合法维度范围内。
- x中所有Tensor数据类型必须一致。
- 属性N指定输入数量，仅用于描述，不影响运行时动态输入列表。

## 调用说明

| 调用方式 | 调用样例                                                    | 说明                                                           |
|--------------|---------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_concat_v2](./examples/test_geir_concat_v2.cpp) | 通过[算子IR](./op_graph/concat_v2_proto.h)构图方式调用ConcatV2算子。 |
