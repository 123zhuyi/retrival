# AccumulateNv2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：实现一组向量的累加。

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
<td>动态输入</td>
<td>待进行累加计算的入参，包含一组向量。</td>
<td>FLOAT、FLOAT16、INT32、INT8、UINT8</td>
<td>ND</td>
</tr>  
<tr>
<td>y</td>
<td>必要输出</td>
<td>累加之后得到的结果。</td>
<td>FLOAT、FLOAT16、INT32、INT8、UINT8</td>
<td>ND</td>
</tr>
<tr>
<td>num</td>
<td>可选属性</td>
<td>一组向量的个数</td>
<td>INT</td>
<td>/</td>
</tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sum.cpp](./examples/test_aclnn_sum.cpp) | 通过[test_aclnn_sum](./docs/aclnnSum.md)接口方式调用AccumulateNv2算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Nice_try | 个人开发者 | AccumulateNv2 | 2026/02/27 | AccumulateNv2算子适配开源仓 |
