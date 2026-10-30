# ReduceVar

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                             |    ×     |

## 功能说明

- 算子功能：返回输入Tensor指定维度的值求得的方差。
- 计算公式：假设dim为 $i$，则对该维度进行计算。$N$为该维度的shape。取 $self_{i}$，求出该维度上的平均值 $\bar{x_{i}}$。

  $$
  out = \frac{1}{max(0, N - \delta N)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2
  $$

  `unbiased = true`时，需加入无偏估计，$\delta N = 1$；`unbiased = false`时，需加入无偏估计，$\delta N = 0$；
  当`keepdim = true`时，reduce后保留该维度，且输出shape中该维度值为1；当`keepdim = false`时，不保留该维度。
  当dim为nullptr或[]时，视为计算所有维度。

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
      <td>待进行ReduceVar计算的输入Tensor，公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>参与Reduce计算的维度，公式中的i。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>unbiased</td>
      <td>输入</td>
      <td>计算方差时是否进行无偏估计，公式中的delta。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>输入</td>
      <td>是否在输出张量中保留输入张量的维度。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>ReduceVar计算的出参，公式中的out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_reduce_var.cpp](./examples/test_aclnn_reduce_var.cpp) | 通过[aclnnVar](../reduce_std_v2/docs/aclnnStd.md)接口方式调用ReduceVar算子。 |
