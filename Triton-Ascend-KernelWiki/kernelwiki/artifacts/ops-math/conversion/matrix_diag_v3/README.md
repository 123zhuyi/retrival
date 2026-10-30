# MatrixDiagV3

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

- 算子功能：根据输入的单条或多条对角线值生成矩阵，对角线带之外的位置使用$\mathrm{padding\_value}$填充。
- 设输出张量最后两维大小分别为$\mathrm{num\_rows}$和$\mathrm{num\_cols}$，$k=[k_l, k_u]$表示待写入的对角线范围，单对角线场景下$k_l = k_u$。令$d = j - i$表示位置$(i, j)$所在的对角线编号，则最大对角线长度为：

$$
\mathrm{max\_diag\_len} = \min(\mathrm{num\_rows} + \min(k_u, 0),\ \mathrm{num\_cols} - \max(k_l, 0))
$$

  输出元素满足：

  $$
  y_{..., i, j} =
  \begin{cases}
  x_{..., k_u - d,\ p(i, j)} & k_l \le d \le k_u \\
  \text{padding\_value} & \text{otherwise}
  \end{cases}
  $$

  其中，$p(i, j)$表示对角线元素在输入$x$最后一维中的位置，具体由属性`align`控制左右对齐方式。当$k$为单个整数或$k_l = k_u$时，$x$表示单条对角线；当$k_l < k_u$时，$x$表示对角线带。

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
      <td>公式中的`x`。当`k`表示单条对角线时，`x`的最后一维保存该对角线的数据；当`k`表示对角线带时，`x`的倒数第二维保存对角线条数，最后一维保存各对角线按`align`补齐后的数据。</td>
      <td>DOUBLE、FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>公式中的k<sub>l</sub>和k<sub>u</sub>。可以是标量，表示单条对角线；也可以是长度为2的向量，表示对角线带的下界和上界。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_rows</td>
      <td>输入</td>
      <td>输出矩阵的行数，即公式中的`num_rows`。取值为`-1`时表示由`k`和`x`自动推导。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_cols</td>
      <td>输入</td>
      <td>输出矩阵的列数，即公式中的`num_cols`。取值为`-1`时表示由`k`和`x`自动推导。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding_value</td>
      <td>输入</td>
      <td>公式中的`padding_value`。用于填充不在指定对角线带内的位置，数据类型与`x`一致。</td>
      <td>DOUBLE、FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128、BOOL</td>
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
      <td>公式中的`y`，表示生成后的矩阵张量，数据类型与`x`一致。当`k`为单个元素或`k[0] == k[1]`时，`y`的秩为`x`的秩加1；否则`y`的秩与`x`一致。</td>
      <td>DOUBLE、FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`的秩至少为1。
- `k`的元素个数只能为1或2；当`k`为2个元素时，必须满足`k[0] <= k[1]`。
- 当`k`表示对角线带时，`x`的倒数第二维长度必须等于$k_u - k_l + 1$，最后一维长度必须等于$\mathrm{max\_diag\_len}$。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_matrix_diag_v3](./examples/test_geir_matrix_diag_v3.cpp) | 通过[算子IR](./op_graph/matrix_diag_v3_proto.h)构图方式调用MatrixDiagV3算子。 |
