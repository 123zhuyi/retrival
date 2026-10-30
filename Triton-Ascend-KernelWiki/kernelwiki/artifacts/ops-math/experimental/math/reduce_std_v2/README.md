# ReduceStdV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>           |  ×   |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：计算指定维度(dim)的标准差和均值，这个dim可以是单个维度、维度列表或者None。
- 计算公式：
  假设 dim 为 $i$，则对该维度进行计算。$N$为该维度的 shape。取 $x_{i}$，求出该维度上的平均值 $\bar{x_{i}}$。
  
  $$
  std = \sqrt{\frac{1}{max(0, N - \delta N)}\sum_{j=0}^{N-1}(x_{ij}-\bar{x_{i}})^2}
  $$
  
  当 `keepdim = true`时，reduce 后保留该维度，且输出 shape 中该维度值为 1；当 `keepdim = false`时，不保留。

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
      <td>待进行ReduceStdV2计算的输入Tensor，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>参与Reduce计算的维度，公式中的i。默认值为空列表，表示对所有维度做Reduce。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>correction</td>
      <td>属性</td>
      <td>修正值，公式中的delta。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>属性</td>
      <td>是否在输出张量中保留输入张量的维度。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_mean_out</td>
      <td>属性</td>
      <td>是否输出均值。true表示输出mean，false表示不输出mean。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>std</td>
      <td>输出</td>
      <td>ReduceStdV2计算的标准差出参，公式中的std。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>ReduceStdV2计算的均值出参，公式中的平均值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

- `dim`中的每个维度值必须在`[-rank(x), rank(x))`范围内，支持负数维度；同一维度归一化后不能重复。
- 当`dim`为空列表时，对`x`的所有维度进行Reduce。
- `std`和`mean`的shape由`dim`和`keepdim`推导：`keepdim = true`时保留被Reduce维度且维度值为1，`keepdim = false`时删除被Reduce维度。
- aclnnStd接口中`out`的shape必须与`dim`和`keepdim`推导结果一致。
- aclnnStd接口中`x`和`out`的数据类型均需在支持列表内，且`x`的数据类型需支持转换为`out`的数据类型；ReduceStdV2图算子的`std`、`mean`与`x`支持相同的数据类型范围。
- 空Tensor场景返回NaN；当被Reduce维度的元素数`shapeProd`为1且`shapeProd <= correction`时返回NaN；当`correction > 1`且`shapeProd <= correction`时返回Inf。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_std.cpp](./examples/test_aclnn_std.cpp) | 通过aclnnStd接口方式调用ReduceStdV2算子。 |
