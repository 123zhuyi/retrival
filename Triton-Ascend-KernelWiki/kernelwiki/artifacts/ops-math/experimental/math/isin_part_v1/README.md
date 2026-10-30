# IsinPart

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：isin功能核心part，结合cat、sort实现isin功能

- isin算子功能：判断给定张量elements的每个元素是否在另一个张量test_elements中出现，并返回一个布尔类型张量，形状与给定张量一致。

- isin算子完整实现：

$$
elementsNum = elements.length() \\
combine\_elements = cat(elements, test\_elements) \\
value, index = sort(combine\_elements) \\
z = isin\_part\_v1(value, index, elementsNum) \\
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px">
<colgroup>
<col style="width: 100px">
<col style="width: 150px">
<col style="width: 410px">
<col style="width: 200px">
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
<td>value</td>
<td>输入</td>
<td>elements与test_elements拼接排序后的张量值</td>
<td>FLOAT、INT32</td>
<td>ND</td>
</tr>

<tr>
<td>index</td>
<td>输入</td>
<td>elements与test_elements拼接排序后的张量索引</td>
<td>INT32</td>
<td>ND</td>
</tr>

<tr>
<td>elementsNum</td>
<td>输入</td>
<td>elements元素数量(标量)</td>
<td>INT64、INT32</td>
<td>ND</td>
</tr>

<tr>
<td>z</td>
<td>输出</td>
<td>isin操作后的输出张量，形状由elementsNum参数决定</td>
<td>BOOL</td>
<td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_isin_part_v1.cpp](./examples/test_aclnn_isin_part_v1.cpp) | 通过[test_aclnn_isin_part_v1]接口方式调用IsinPartV1算子。 |
