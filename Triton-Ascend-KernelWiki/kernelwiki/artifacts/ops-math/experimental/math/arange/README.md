# Arange

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：从start起始到end结束按照step的间隔取值，并返回大小为 $\frac{end-start}{step}+1$的1维张量。其中，步长step是张量中相邻两个值的间隔。

- 计算公式：

$$
\text{out}_{i+1} = \text{out}_i + \text{step}
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
      <td>start</td>
      <td>输入</td>
      <td>Host侧的aclScalar，获取值的范围的起始位置。</td>
      <td>FLOAT16、FLOAT、INT32、INT64、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>输入</td>
      <td>Host侧的aclScalar，获取值的范围的结束位置。</td>
      <td>FLOAT16、FLOAT、INT32、INT64、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>step</td>
      <td>输入</td>
      <td>Host侧的aclScalar，获取值的步长。</td>
      <td>FLOAT16、FLOAT、INT32、INT64、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行Arange计算的出参。</td>
      <td>FLOAT16、FLOAT、INT32、INT64、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 需要满足在step大于0时输入的start小于end，或者step小于0时输入的start大于end。
- 需要满足step不等于0。
- start、end、step、out的数据类型只支持FLOAT16、FLOAT、INT32、INT64、BFLOAT16，数据格式只支持ND

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_arange](./examples/test_aclnn_arange.cpp) | 通过aclnnArange接口方式调用Arange算子。 |
