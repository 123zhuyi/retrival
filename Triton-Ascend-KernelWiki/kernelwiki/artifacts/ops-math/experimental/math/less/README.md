# Less

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

- 算子功能：判断输入self中的每个元素是否小于输入other的值，返回一个Bool类型的Tensor。

- 计算公式：

  $$
  y_i = (x1_i < x2_i)  ?  [True] : [False]
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入张量x1。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入张量x2。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                             | 说明           |
|--------------|------------------------|--------------------------|
| aclnn调用 | [test_aclnn_lt_scalar](./examples/test_aclnn_lt_scalar.cpp) | 通过[aclnnLtScalar&aclnnInplaceLtScalar](./docs/aclnnLtScalar&aclnnInplaceLtScalar.md)接口方式调用Less算子。    |
| aclnn调用 | [test_aclnn_lt_tensor](./examples/test_aclnn_lt_tensor.cpp) | 通过[aclnnLtTensor&aclnnInplaceLtTensor](./docs/aclnnLtTensor&aclnnInplaceLtTensor.md)接口方式调用Less算子。    |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| fulltower | 个人开发者 | Less | 2026/2/27 | Less算子适配开源仓 |
| 浙江工业大学杨旭华团队 | 个人开发者 | Less | 2025/7/30 | 新增Less算子 |
