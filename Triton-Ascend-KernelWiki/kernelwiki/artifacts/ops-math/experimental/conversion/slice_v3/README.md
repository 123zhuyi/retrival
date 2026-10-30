# Slice

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：从输入张量中提取一个切片。切片从begin位置开始，以size大小结束。支持任意维度的切片。

- 计算公式：
给定输入张量 x，起始位置 begin 和切片大小 size，输出张量 y 的计算公式为：

$$
y[i_1, i_2, \ldots, i_n] = x[\text{begin}[0] + i_1, \text{begin}[1] + i_2, \ldots, \text{begin}[n] + i_n]
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px">
<colgroup>
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
<td>待进行切片操作的输入张量</td>
<td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16、INT8、UINT8</td>
<td>ND</td>
</tr>

<tr>
<td>begin</td>
<td>输入</td>
<td>每个维度的起始位置，长度为输入张量的维度数</td>
<td>INT64</td>
<td>ND（一维张量）</td>
</tr>

<tr>
<td>size</td>
<td>输入</td>
<td>每个维度的切片大小，长度为输入张量的维度数</td>
<td>INT64</td>
<td>ND（一维张量）</td>
</tr>

<tr>
<td>y</td>
<td>输出</td>
<td>切片操作后的输出张量，形状由size参数决定</td>
<td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16、INT8、UINT8</td>
<td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_slice_v3.cpp](./examples/test_aclnn_slice_v3.cpp) | 通过[test_aclnn_slice_v3]接口方式调用SliceV3算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| cc | 哈尔滨工业大学-苏统华团队 | Slice | 2025/12/17 | 新增Slice算子 |
