# Flatten

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    √     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- **算子功能**：将输入Tensor基于给定的axis扁平化为一个2D的Tensor。
- **计算公式**：
  - 若输入self的shape为(d₀, d₁, ..., dₙ)，则输出out的shape为(d₀×d₁×...×dₐₓᵢₛ₋₁, dₐₓᵢₛ×dₐₓᵢₛ₊₁×...×dₙ)
  - 若axis取值为0，则输出out的shape为(1, d₀×d₁×...×dₙ)

## 参数说明

<table style="undefined;table-layout: fixed; width: 949px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 202px">
<col style="width: 335px">
<col style="width: 102px">
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
    <td>输入张量</td>
    <td>输入张量，支持非连续的Tensor，2D~8D。</td>
    <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16、FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>axis</td>
    <td>输入属性</td>
    <td>flatten计算的基准轴，取值范围为[-self.dim(), self.dim()-1]，目前仅支持axis = 1。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>输出2D张量，支持非连续的Tensor。</td>
    <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16、FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入张量的维度不能大于8。
- axis必须在有效的维度范围内。
- 输入和输出的数据类型必须一致。

## 调用说明

| 调用方式  | 样例代码                                                | 说明                                                         |
| --------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_flatten](./examples/test_aclnn_flatten.cpp) | 通过[aclnnFlatten](docs/aclnnFlatten.md)接口方式调用Flatten算子。 |
