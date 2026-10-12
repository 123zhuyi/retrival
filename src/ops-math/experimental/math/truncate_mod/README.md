# TruncateMod

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：逐元素计算截断除法的余数，余数与被除数 `x1` 同号。

- 计算公式：

  对每个位置 `i`，先计算截断商（向零取整）：

  $$
  tq_i = trunc(x1_i / x2_i) = \lfloor \max(x1_i / x2_i, 0) \rfloor + \lceil \min(x1_i / x2_i, 0) \rceil
  $$

  再计算余数：

  $$
  y_i = x1_i - tq_i \cdot x2_i
  $$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>公式中的输入x1，被除数。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2，除数，shape 与 x1 一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y，截断除法余数，数据类型与输入一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x1`、`x2`、`y` 的 shape 需保持一致，不支持广播。
- `x1`、`x2` 的数据类型需保持一致，输出 `y` 与输入同类型。
- 除零为未定义行为。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_truncate_mod](./examples/test_aclnn_truncate_mod.cpp) | 通过[aclnnTruncateMod](./docs/aclnnTruncateMod.md)接口方式调用TruncateMod算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| weixin_45448057 | 个人开发者 | TruncateMod | 2026/07/14 | TruncateMod算子适配开源仓 |
