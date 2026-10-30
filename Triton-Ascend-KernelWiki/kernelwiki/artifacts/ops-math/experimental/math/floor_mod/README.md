# FloorMod

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :----:|
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：对输入张量进行逐元素取模运算，结果的符号与除数（`x2`）一致（向下取整）。

- 计算公式：

$$
y_i = x1_i - \lfloor \frac{x1_i}{x2_i} \rfloor \times x2_i
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
      <td>x1</td>
      <td>输入</td>
      <td>待进行FloorMod计算的被除数入参。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行FloorMod计算的除数入参。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>FloorMod计算的输出结果。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x1` 和 `x2` 的 shape 需要满足广播（Broadcast）规则。
- 数据类型需要一致，或者满足隐式类型转换规则。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| aclnn调用 | [test_aclnn_floor_mod.cpp](./examples/test_aclnn_floor_mod.cpp) | 通过[aclnnRemainder](./op_api/aclnn_remainder.h)接口方式调用FloorMod算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Tream | 个人开发者 | FloorMod | 2025/12/17 | FloorMod算子适配开源仓 |
