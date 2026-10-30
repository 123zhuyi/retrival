# MatrixDiagPartV3

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：从输入矩阵或批量矩阵的最后两维中提取一条或多条对角线，并按照`align`指定的方式使用$\mathrm{padding\_value}$对较短对角线进行补齐。
- 设输入$x$的最后两维大小为$M$和$N$，$k=[k_l, k_u]$表示待提取的对角线范围。令：

  $$
  \mathrm{num\_diags} = k_u - k_l + 1
  $$

  $$
  \mathrm{max\_diag\_len} = \min(M + \min(k_u, 0),\ N - \max(k_l, 0))
  $$

  对于编号为$d$的对角线，其有效长度为：

  $$
  \mathrm{diag\_len}(d) = \min(M + \min(d, 0),\ N - \max(d, 0))
  $$

  输出元素满足：

  $$
  y_{..., m, n} =
  \begin{cases}
  x_{..., n + \max(-d, 0),\ n + \max(d, 0)} & 0 \le n < \mathrm{diag\_len}(d) \\
  \text{padding\_value} & \text{otherwise}
  \end{cases}
  $$

  其中，带状输出场景下$d = k_u - m$；单对角线场景下$d = k_l = k_u$。当$k$为单个整数或$k_l = k_u$时，输出shape为`[..., max_diag_len]`；当$k_l < k_u$时，输出shape为`[..., num_diags, max_diag_len]`。

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
      <td>公式中的`x`。最后两维表示待提取对角线的矩阵，其余前置维度按batch维处理。</td>
      <td>DOUBLE、FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>公式中的k<sub>l</sub>和k<sub>u</sub>。可以是标量，表示提取单条对角线；也可以是长度为2的向量，表示提取对角线带。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding_value</td>
      <td>输入</td>
      <td>公式中的`padding_value`。用于补齐较短对角线的无效位置，数据类型与`x`一致。</td>
      <td>DOUBLE、FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>align</td>
      <td>可选属性</td>
      <td>指定超对角线和次对角线的对齐方式。支持`RIGHT_LEFT`、`LEFT_RIGHT`、`LEFT_LEFT`、`RIGHT_RIGHT`，默认值为`RIGHT_LEFT`。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的`y`，表示提取后的对角线Tensor。数据类型与`x`一致。</td>
      <td>DOUBLE、FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`的秩至少为2。
- `k`的元素个数只能为1或2；当`k`为2个元素时，必须满足`k[0] <= k[1]`。
- 当`k`表示对角线带时，输出的倒数第二维长度为$k_u - k_l + 1$，最后一维长度为$\mathrm{max\_diag\_len}$。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_matrix_diag_part_v3](./examples/test_geir_matrix_diag_part_v3.cpp) | 通过[算子IR](./op_graph/matrix_diag_part_v3_proto.h)构图方式调用MatrixDiagPartV3算子。 |
