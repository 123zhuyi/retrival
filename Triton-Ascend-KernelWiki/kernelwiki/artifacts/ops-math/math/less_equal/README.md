# LessEqual

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：判断输入self中的元素值是否小于等于other的值，并将self的每个元素的值与other值的比较结果写入out中。
- 计算公式：

  $$
  out_i = (self_i <= other) ? [True] : [False]
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
      <td>公式中的输入张量self_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、UINT64、UINT8、INT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入张量other。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、UINT64、UINT8、INT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量out_i。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_le_scalar](./examples/test_aclnn_le_scalar.cpp) | 通过[aclnnLeScalar&aclnnInplaceLeScalar](./docs/aclnnLeScalar&aclnnInplaceLeScalar.md)接口方式调用LessEqual算子。    |
