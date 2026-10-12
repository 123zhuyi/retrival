# MatrixSetDiagV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|×|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|×|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|

## 功能说明

- 算子功能：将输入tensor的对角线元素替换为对角线tensor的值。

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
      <td>input</td>
      <td>输入</td>
      <td>待进行替换的原始tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX64、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>输入</td>
      <td>对角线tensor。</td>
      <td>与input的数据类型保持一致。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>对角线的取值范围。</td>
      <td>INT32。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>进行替换后的tensor。</td>
      <td>与input的数据类型保持一致。</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `input`的维度最少为2维，最大不超过8维。
- `k`是数据类型INT32的标量或长度为2的向量。
- 当`k`为长度2的向量时，需满足`k[1] >= k[0]`。
- 当`k`为标量或`k[0] = k[1]`（单对角线）时，`diagonal`的维度比`input`小1维：最后一维的长度为`maxDiagLen = min(row + min(k[1], 0), col - max(k[0], 0))`，其中`row`、`col`分别为`input`最后两维的长度；其余维度与`input`的对应维度一一相等。
- 当`k`不是标量且`k[0] != k[1]`（多对角线）时，`diagonal`的维度与`input`相同：最后两维的长度分别为`numDiags = k[1] - k[0] + 1`与`maxDiagLen = min(row + min(k[1], 0), col - max(k[0], 0))`，其中`row`、`col`分别为`input`最后两维的长度，`numDiags`为对角线数量；除最后两维外，其余维度与`input`的对应维度一一相等。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 图模式调用 | [test_geir_matrix_set_diag_v2](examples/test_geir_matrix_set_diag_v2.cpp) | 通过[算子IR](op_graph/matrix_set_diag_v2_proto.h)构图方式调用MatrixSetDiagV2算子。 |
