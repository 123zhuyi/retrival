# GetShape

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √   |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √   |
| <term>Atlas 训练系列产品</term>                              |    √    |

## 功能说明

- 算子功能：获取一个或多个输入 tensor 的 shape 信息，按输入顺序将各维度的尺寸值拼接为一维 int32 tensor 输出。

- 计算逻辑：
  - 输入 N 个 tensor，第 i 个 tensor 的 rank 为 r_i
  - 输出为一维 int32 tensor，长度为 sum(r_i)，内容为所有输入 tensor 各维度值的顺序拼接

- 示例：
  ```
  输入: x0 = float32[2, 3, 4], x1 = float32[5, 6]
  输出: y = int32[5] = [2, 3, 4, 5, 6]
  ```

## 参数说明

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>动态输入</td>
      <td>待获取 shape 信息的输入 tensor 列表。</td>
      <td>DOUBLE、FLOAT、FLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>所有输入 tensor 各维度值的顺序拼接，长度为各输入 rank 之和。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 每个输入 tensor 的 rank 不超过 8。
- 所有输入 tensor 的 rank 之和（即输出长度）不超过 128，且大于 0（不允许所有输入均为 scalar）。
- 输出 dtype 固定为 INT32，不可配置。
- 输入仅支持 ND 格式。

## 调用说明

| 调用方式 | 调用样例                                              | 说明                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| 图模式调用 | [test_geir_get_shape](./examples/test_geir_get_shape.cpp)   | 通过[算子IR](./op_graph/get_shape_proto.h)构图方式调用GetShape算子。                   |
