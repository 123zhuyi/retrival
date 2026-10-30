# Reshape

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：在不改变底层数据布局的前提下，将输入张量重解释为目标形状。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
 <col style="width: 140px">
 <col style="width: 180px">
 <col style="width: 360px">
 <col style="width: 500px">
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
   <td>待重塑的输入张量。</td>
   <td>BOOL, FLOAT, FLOAT16, INT8, INT16, UINT16, UINT8, INT32, INT64, UINT32, UINT64, DOUBLE, COMPLEX64, COMPLEX128, BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shape</td>
   <td>输入</td>
   <td>目标形状张量，支持一个 <code>-1</code>；默认模式下 <code>0</code> 表示复制对应输入维度。</td>
   <td>INT32, INT64</td>
   <td>1D</td>
  </tr>
  <tr>
   <td>axis</td>
   <td>属性</td>
   <td>起始替换维度，默认值为0。</td>
   <td>INT</td>
   <td>-</td>
  </tr>
  <tr>
   <td>num_axes</td>
   <td>属性</td>
   <td>被替换的连续维度数，默认值为 -1。</td>
   <td>INT</td>
   <td>-</td>
  </tr>
  <tr>
   <td>y</td>
   <td>输出</td>
   <td>重塑后的输出张量，数据类型与输入x相同。</td>
   <td>与x相同</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## 约束说明

- 输出元素总数必须与输入元素总数一致。
- shape中最多只能出现一个 <code>-1</code>。
- 当allowzero未设置或为0时，shape中的 <code>0</code> 会复制对应输入维度；当allowzero为1时，<code>0</code> 按字面值参与计算。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_reshape](./examples/test_geir_reshape.cpp) | 通过 [算子IR](./op_graph/reshape_proto.h)构图方式调用reshape算子。 |
