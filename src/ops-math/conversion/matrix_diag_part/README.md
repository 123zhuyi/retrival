# MatrixDiagPart

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                             |    ×     |

## 功能说明

- 算子功能：返回输入矩阵的最内层矩阵的主对角线（k=0）。

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
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待计算主对角线的入参，最后两个维度为矩阵维度。</td>
      <td>float16、float32、int32、int8、uint8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>主对角线结果，与x同dtype。</td>
      <td>float16、float32、int32、int8、uint8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x的维度数需大于等于2且小于等于8，最后两个维度表示矩阵。
- 输出y的维度数比输入x少1，形状为x.shape[:-2]+[min(M, N)]。
- 支持的数据类型：float16、float32、int32、int8、uint8。
- 数据格式仅支持ND。
- 仅支持连续输入，不支持非连续输入。
- 支持空tensor（M=0或N=0时，输出为空tensor）。
- 该算子为MatrixDiagPart V1版本，仅提取主对角线（k=0）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| 图模式 | [test_geir_matrix_diag_part](examples/test_geir_matrix_diag_part.cpp) | 通过[算子IR](op_graph/matrix_diag_part_proto.h)构图方式调用MatrixDiagPart算子，参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成编译和验证。 |
