# ConjugateTranspose

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：按perm指定的维度顺序对输入x做转置，并对结果取共轭。对实数类型，共轭等价于恒等，等价于普通Transpose。

- 计算公式：y = conj(transpose(x, perm))，其中y的第i维大小为x的第perm[i]维大小。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>待转置的输入张量，维度需大于1。</td>
      <td>COMPLEX64、COMPLEX128、FLOAT16、FLOAT、DOUBLE、BOOL、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>perm</td>
      <td>输入</td>
      <td>1-D张量，描述维度置换顺序，元素个数等于x的维数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>转置并取共轭后的张量，类型与x一致。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- x的维数需大于1，且不超过7维。
- perm必须为1-D，元素个数等于x的维数，取值为x各维度的一个排列。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_conjugate_transpose](./examples/test_geir_conjugate_transpose.cpp)   | 通过[算子IR](./op_graph/conjugate_transpose_proto.h)构图方式调用conjugate_transpose算子。 |
