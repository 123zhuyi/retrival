# Fill

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：实现张量的填充功能，将张量中的所有元素填充为指定的统一值。

- 计算公式：
对于任意索引位置 `(i, j, ...)`，执行 `fill(T, v)` 后：

$$
\forall (i, j, \dots), \quad T[i, j, \dots] = v
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
    </tr>
  </thead>
  <tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待进行fill算子计算的入参，公式中的T</td>
    <td>FLOAT16、FLOAT、INT16、INT32、BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fill_value</td>
    <td>输入</td>
    <td>用于填充的标量值，公式中的v</td>
    <td>FLOAT16、FLOAT、INT16、INT32、BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x</td>
    <td>输出</td>
    <td>执行fill算子计算后的出参，公式中的T'</td>
    <td>FLOAT16、FLOAT、INT16、INT32、BF16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fill_v2.cpp](./examples/test_aclnn_fill_v2.cpp) | 通过[test_aclnn_s_where]接口方式调用Fill算子。 |
