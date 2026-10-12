# TruncateDiv

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：完成截断除法计算，对输入x1和x2逐元素相除，结果向零取整（截断取整）。

- 计算公式：

$$
y = trunc(\frac{x1}{x2})
$$

其中 $trunc$ 表示向零取整。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>待进行TruncateDiv计算的入参，公式中的被除数x1。</td>
      <td>bfloat16,float16,float,int32,uint8,int8,int64,int16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行TruncateDiv计算的入参，公式中的除数x2。</td>
      <td>bfloat16,float16,float,int32,uint8,int8,int64,int16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行TruncateDiv计算的出参，公式中的输出y。</td>
      <td>bfloat16,float16,float,int32,uint8,int8,int64,int16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 当除数x2为0时，结果为未定义行为。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_truncate_div](./examples/test_aclnn_truncate_div.cpp) | 通过aclnnTruncateDiv接口方式调用TruncateDiv算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| weixin_45448057 | 个人开发者 | TruncateDiv | 2026/07/14 | TruncateDiv算子适配开源仓 |
