# ConfusionTransposeD

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|

## 功能说明

- 算子功能：融合reshape和transpose运算。

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
      <td>输入</td>
      <td>输入张量。</td>
      <td>INT8、INT16、 INT32、 INT64、UINT8、UINT16、UINT32、UINT64、FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>perm</td>
      <td>输入</td>
      <td>转置后每根轴对应的转置前轴索引。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>输入</td>
      <td>reshape后的shape大小。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transposeFirst</td>
      <td>输入</td>
      <td>判断是否先执行transpose操作。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>与输入x保持一致。</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

-perm张量中元素必须唯一，并在[0,perm的维度数量-1]范围内。
-当transposeFirst为True时，perm的长度必须与x的shape的长度相同，即len(perm)=len(x_shape)；当transposeFirst为False时，perm长度必须与属性输入shape的长度相同，即len(perm)=len(shape)。
-shape中的所有维度乘积必须等于输入张量x的元素总数。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_confusion_transpose.cpp](./examples/test_aclnn_confusion_transpose.cpp) | 通过[aclnnConfusionTranspose](./docs/aclnnConfusionTranspose.md)接口方式调用ConfusionTransposeD算子。 |
