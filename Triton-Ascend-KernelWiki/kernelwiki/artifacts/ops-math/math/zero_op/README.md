# ZerosLike

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |   √   |
| <term>Atlas 训练系列产品</term>                              |   √   |

## 功能说明

* 算子功能：将selfRef张量填充为全零。

- 计算公式：

  $$
  {selfRef}_{i} = 0
  $$

- 示例：

  ```text
  输入selfRef：
  tensor([[1, 2],
          [3, 4]])
  输出selfRef：
  tensor([[0, 0],
          [0, 0]])
  ```

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
      <td>待进行zeroslike计算的入参，公式中的selfRef。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行zeroslike计算的出参，将selfRef张量填充为全0。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_zero](./examples/test_aclnn_zero.cpp) | 通过[aclnnInplaceZero](./docs/aclnnInplaceZero.md)接口方式调用ZerosLike算子。    |
