# Triu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×     |
| <term>Atlas 推理系列产品</term>                             |   √     |
| <term>Atlas 训练系列产品</term>                              |   √     |

## 功能说明

- 算子功能：将输入的self张量的最后二维（按shape从左向右数）沿对角线的左下部分置零。参数diagonal可正可负，默认为零，正数表示主对角线向右上方向移动，负数表示主对角线向左下方向移动。
- 计算公式：下面用i表示遍历倒数第二维元素的序号（i是行索引），用j表示遍历最后一维元素的序号（j是列索引），用d表示diagonal，在(i, j)对应的二维坐标图中，i+d==j表示在对角线上。

  $$
  对角线及其右上方，即i+d<=j，保留原值： out_{i, j} = self_{i, j}\\
  而位于对角线左下方的情况，即i+d>j，置零（不含对角线）：out_{i, j} = 0
  $$

- 示例：

  $self = \begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$，
  triu(self, diagonal=0)的结果为：
  $\begin{bmatrix} [9&6&3] \\ [0&2&3] \\ [0&0&1] \end{bmatrix}$；
  调整diagonal的值，triu(self, diagonal=1)结果为：
  $\begin{bmatrix} [0&6&3] \\ [0&0&3] \\ [0&0&0] \end{bmatrix}$；
  调整diagonal为-1，triu(self, diagonal=-1)结果为：
  $\begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [0&4&1] \end{bmatrix}$。

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
      <td>待进行triu计算的入参，公式中的self。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>输入</td>
      <td>待进行triu计算的入参，公式中的diagonal。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行triu计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_triu](./examples/test_aclnn_triu.cpp) | 通过[aclnnTriu](./docs/aclnnTriu&aclnnInplaceTriu.md)接口方式调用Triu算子。 |
