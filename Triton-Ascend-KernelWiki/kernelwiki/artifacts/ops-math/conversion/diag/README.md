# Diag

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：将输入tensor(展平视为1D)的对角线元素，展开为2D对角矩阵

- 计算公式：

    设：
    $$
    \mathbf{x} \in \mathbb{R}^n ：输入向量
    $$
    $$
    \mathbf{y} \in \mathbb{R}^{n \times n}：输出对角矩阵
    $$

    其中：
    $$
    \mathbf{y}_{i,i} = x_i
    $$
    $$
    \mathbf{y}_{i,j} = 0 当i \ne j
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的x。</td>
      <td>INT64、INT32、FLOAT、FLOAT16、DOUBLE、BF16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>INT64、INT32、FLOAT、FLOAT16、DOUBLE、BF16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持BF16。

## 约束说明

只支持输入shape维度1-4维，对应输出2-8维，不支持标量输入。

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| 图模式调用 | NA  | 通过[算子IR](./op_graph/diag_proto.h)构图方式调用diag算子
