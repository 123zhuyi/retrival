# Cross

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |

## 功能说明

- 算子功能：对输入Tensor完成linear_cross运算。

- 计算公式：

$$
out = self\times other = \begin{vmatrix}i&j&k\\x_1&y_1&z_1\\x_2&y_2&z_2\end{vmatrix} = (y_1z_2-y_2z_1)i-(x_1z_2-x_2z_1)j+(x_1y_2-x_2y_1)k
$$

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>计算公式中的输入 self。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8、INT16、DOUBLE、INT64、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>计算公式中的输入 other。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8、INT16、DOUBLE、INT64、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算公式中的输出 out。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8、INT16、DOUBLE、INT64、UINT16、UINT32、UINT64、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持INT8、INT16、INT32、INT64、UINT8、FLOAT16、BFLOAT16、FLOAT、FLOAT64、COMPLEX64、COMPLEX128。

## 约束说明

- 指定维度（进行cross运算的维度）的size必须为3。
- 输入shape需满足broadcast关系，即两个输入tensor的shape需要兼容，能够进行广播运算。
- 输入数据类型需要保持一致。

## 调用说明

| 调用方式 | 调用样例                                                    | 说明                                                               |
|--------------|---------------------------------------------------------|------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_linalg_cross.cpp](examples/test_aclnn_linalg_cross.cpp) | 通过[aclnnLinalgCross](docs/aclnnLinalgCross.md)接口方式调用Cross算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| hth810 | 个人开发者 | Cross | 2026/4/9 | Cross算子适配开源仓 |
