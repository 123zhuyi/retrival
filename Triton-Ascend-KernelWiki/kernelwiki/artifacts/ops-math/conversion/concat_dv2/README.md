# ConcatDV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|x|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|x|
|Atlas 推理系列产品|x|
|Atlas 训练系列产品|x|

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
       y = ConcatDV2(x[0], x[1], ..., x[N-1], axis = concat_dim)
  - 输出：拼接后的Tensor y

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 930px"><colgroup>
  <col style="width: 100px">
  <col style="width: 140px">
  <col style="width: 270px">
  <col style="width: 320px">
  <col style="width: 100px">
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
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT32、UINT8、INT16、INT8、COMPLEX64、INT64、BOOL</td>
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
      <td>y</td>
      <td>输出</td>
      <td>计算流程中的输出y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT32、UINT8、INT16、INT8、COMPLEX64、INT64、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入Tensor在除拼接维度外的形状必须一致。
- 拼接维度concat_dim当前仅支持为0的情况。
- x中所有Tensor数据类型必须一致。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_cat](../concat_d/examples/test_aclnn_cat.cpp) | 通过[aclnnCat](../concat_d/docs/aclnnCat.md)接口方式调用ConcatDV2算子。 |
