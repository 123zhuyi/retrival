# TruncateDiv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：完成截断除法计算，结果向零取整

- 计算公式：

  $$
  out_i = trunc(\frac{x1_i}{x2_i})
  $$

  其中`trunc`表示向零取整（截断取整）。

- 例外说明：无

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的被除数。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、UINT8、INT8、INT64、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的除数。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、UINT8、INT8、INT64、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的out，截断除法结果。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、UINT8、INT8、INT64、INT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x1和x2需满足broadcast关系
- 当除数为0时，结果为未定义行为

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_truncate_div](./examples/test_geir_truncate_div.cpp)   | 通过[算子IR](./op_graph/truncate_div_proto.h)构图方式调用TruncateDiv算子。 |
