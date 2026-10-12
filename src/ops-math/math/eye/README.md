# Eye

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

算子功能：返回一个二维张量，该张量的对角线上元素值为1，其余元素值为0。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>n</td>
      <td>输入</td>
      <td>表示第一个维度，Host侧的整型。取值范围不小于0。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>m</td>
      <td>输入</td>
      <td>表示第二个维度，Host侧的整型。取值范围不小于0。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_shape</td>
      <td>输入</td>
      <td>可选，表示输出张量的batch维度，Host侧的整型列表，每项必须大于0。未指定时输出为二维。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>输入</td>
      <td>可选，表示输出张量的数据类型，Host侧的整型，对应数据类型枚举值。默认值为FLOAT32。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示输出张量，Device侧的aclTensor。shape为batch_shape + (n, m)，未指定batch_shape时为二维(n, m)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32、INT16、INT8、UINT8、INT64、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_eye](./examples/test_aclnn_eye.cpp) | 通过[aclnnEye](./docs/aclnnEye.md)接口方式调用Eye算子。 |
