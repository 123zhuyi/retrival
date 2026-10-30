# MatrixDiag

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

- 算子功能：返回一个给定批量对角值的批量对角Tensor。

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
      <td>待进行MatrixDiag计算的入参，公式中的x。</td>
      <td>DOUBLE, FLOAT32, FLOAT16, BFLOAT16, COMPLEX32, COMPLEX64, COMPLEX128, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, QINT8, QUINT8, QINT16, QUINT16, QINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, BOOL.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行MatrixDiag计算的出参，公式中的out。</td>
      <td>DOUBLE, FLOAT32, FLOAT16, BFLOAT16, COMPLEX32, COMPLEX64, COMPLEX128, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, QINT8, QUINT8, QINT16, QUINT16, QINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, BOOL.</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_matrix_diag](./examples/test_geir_matrix_diag.cpp)   | 通过[算子IR](./op_graph/matrix_diag_proto.h)构图方式调用MatrixDiag算子。
