# SparseReshape

## 贡献说明

| 贡献者      | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------|------------|----------|

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：将稀疏张量(SparseTensor)的indices从输入shape重塑到目标shape。稀疏张量由indices(非零元素坐标)、values(非零元素值)和shape(稠密形状)组成。SparseReshape仅对indices和shape进行重塑，不修改values。

- 计算公式：

$$
\text{flat\_id} = \sum_{j=0}^{\text{input\_rank}-1} \text{indices}[i, j] \times \text{input\_strides}[j]
$$

$$
\text{y\_indices}[i, j] = \lfloor \text{flat\_id} / \text{output\_strides}[j] \rfloor, \quad \text{flat\_id} = \text{flat\_id} \bmod \text{output\_strides}[j]
$$

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
      <td>indices</td>
      <td>输入</td>
      <td>非零元素的多维坐标矩阵，2D张量，shape=(nnz, input_rank)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>输入</td>
      <td>原始稠密形状，1D张量，shape=(input_rank,)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_shape</td>
      <td>输入</td>
      <td>目标稠密形状，1D张量，shape=(output_rank,)。允许一个维度为-1，自动推导。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_indices</td>
      <td>输出</td>
      <td>重塑后的非零元素坐标矩阵，2D张量，shape=(nnz, output_rank)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_shape</td>
      <td>输出</td>
      <td>重塑后的稠密形状，1D张量，shape=(output_rank,)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入indices必须为2D矩阵，shape和new_shape必须为1D向量。
- 输入indices每行是一个非零元素的多维坐标。元素值大于等于0且小于对应shape维度的size。
- input_rank和output_rank均不超过8维(MAX_RANK=8)。
- new_shape中最多允许一个维度为-1，该维度值由总元素数除以其他维度乘积自动推导。
- 输入shape和输出new_shape的总元素数必须一致。
- 所有输入和输出的dtype必须相同(全部int32或全部int64)。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_sparse_reshape.cpp">test_geir_sparse_reshape</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
