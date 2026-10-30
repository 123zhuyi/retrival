# Pow2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：实现张量的指数运算功能，对输入张量 x1（底数）和 x2（指数）进行元素级计算。

- 计算公式：

$$
out_i = \text{pow}(x1_i, x2_i)
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
      <td>底数张量，可为标量或多维张量，支持广播到输出张量。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16、INT8、UINT8、INT16、INT32</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>指数张量，可为标量或多维张量，支持广播到输出张量。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16、INT8、UINT8、INT16、INT32</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>元素级计算结果张量，输出数据类型与输入类型一致或通过 Cast 转换。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16、INT8、UINT8、INT16、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 暂不支持 int64/uint64 类型。
- 输入张量维度可不同，但需要通过广播规则匹配。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_pow2.cpp](./examples/test_aclnn_pow2.cpp) | 通过[test_aclnn_pow2](./docs/test_aclnn_pow2.md)接口方式调用SelectV2算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Shi xiangyang | 个人开发者 | SelectV2 | 2025/12/16 | Pow算子适配开源仓 |
