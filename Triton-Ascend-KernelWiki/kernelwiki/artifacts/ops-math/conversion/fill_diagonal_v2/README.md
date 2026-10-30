# FillDiagonalV2

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    ×     |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：以fillValue填充tensor对角线。
- 计算公式：
  - 以二维为例，`wrap`为False时，填充位置为`[r, r]`，其中`0 <= r < m`，`m = min(col, row)`，`col`为列的长度，`row`为行的长度。
  - `wrap`为True时，填充位置为`[r + (m + 1) * i , r]`，其中`0 <= r < m`，`m = min(col, row)`，`col`为列的长度，`row`为行的长度，`0 <= i < col // m`。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1043px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 320px">
<col style="width: 96px">
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
    <td>selfRef</td>
    <td>输入/输出张量</td>
    <td>表示输入/输出张量，支持非连续的Tensor。</td>
    <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fillValue</td>
    <td>输入属性</td>
    <td>表示填充值，数据类型需要是可转换为FLOAT的数据类型。</td>
    <td>可转换为FLOAT的数据类型</td>
    <td>-</td>
  </tr>
  <tr>
    <td>wrap</td>
    <td>输入属性</td>
    <td>表示填充方式，对于高矩阵（行数row大于列数col），若wrap值为True，每经过N行形成一条新的对角线，其中N = min(col, row)。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody>
</table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

- selfRef的维度必须大于1。
- 当selfRef的维度大于2时，各维度的长度必须相同。
- fillValue必须能转换为FLOAT类型，并且在转换为selfRef的数据类型时不能发生溢出。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_inplace_fill_diagonal](./examples/test_aclnn_fill_diagonal_v2.cpp) | 通过[aclnnInplaceFillDiagonal](docs/aclnnInplaceFillDiagonal.md)接口方式调用InplaceFillDiagonal算子。 |
