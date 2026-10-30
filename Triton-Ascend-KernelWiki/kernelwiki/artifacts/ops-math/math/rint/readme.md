# Rint

## 产品支持情况

| 产品　　　　　　　　　　　　　　　　　　　　　　　　　　 | 是否支持 |
| :---------------------------------------------------------| :--------:|
| <term>Ascend 950PR/Ascend 950DT</term>　　　　　　　　　 | √　　　　|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √　　　　|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √　　　　|
| <term>Atlas 200I/500 A2 推理产品</term>　　　　　　　　　| ×　　　　|
| <term>Atlas 推理系列产品</term>　　　　　　　　　　　　　| √　　　　|
| <term>Atlas 训练系列产品</term>　　　　　　　　　　　　　| √　　　　|

## 功能说明

- 算子功能：获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。

- 计算公式：

$$
out_{i}=rint(input_{i})
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
      <td>self</td>
      <td>输入</td>
      <td>待进行rint计算的入参，公式中的input_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>进行rint计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式　 | 调用样例　　　　　　　　　　　　　　　　　　　　| 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| -------------------------------------------------| -------------------------------------------------------------|
| 图模式调用 | [test_geir_rint](./examples/test_geir_rint.cpp) | 通过[算子IR](./op_graph/rint_proto.h)构图方式调用Rint算子。 |
