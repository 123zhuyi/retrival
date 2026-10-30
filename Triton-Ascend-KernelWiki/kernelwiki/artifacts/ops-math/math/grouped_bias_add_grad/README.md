# GroupedBiasAddGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对分组通道的偏置梯度进行归约求和。

- 计算公式：

&emsp;&emsp;(1)有可选输入groupIdxOptional，且groupIdxType为0时：

$$
out(G,H) =
\begin{cases}
\displaystyle
\sum_{i = \mathrm{groupIdxOptional}(j-1)}^{\mathrm{groupIdxOptional}(j)}  \!\! \mathrm{gradY}(i, H), & 1 \leq j \leq G-1 \\[8pt]
\displaystyle
\sum_{i = 0}^{\mathrm{groupIdxOptional}(j)}  \mathrm{gradY}(i, H), & j = 0
\end{cases}
$$

&emsp;&emsp;(2)有可选输入groupIdxOptional，且groupIdxType为1时：

$$
groupIdx(i) = \sum_{i=0}^{j} groupIdxOptional(j), j=0...G
$$

$$
out(G,H) =
\left\{
\begin{aligned}
&\sum_{i\,=\,\mathrm{groupIdx}(j-1)}^{\mathrm{groupIdx}(j)} \!\! \mathrm{gradY}(i, H), && 1 \leq j \leq G-1 \\
&\sum_{i\,=\,0}^{\mathrm{groupIdx}(j)} \mathrm{gradY}(i, H), && j = 0
\end{aligned}
\right.
$$

&emsp;&emsp;其中，gradY共2维，H表示gradY最后一维的大小，G表示groupIdxOptional第0维的大小，即groupIdxOptional有G个数，groupIdxOptional(j)表示第j个数的大小，计算后out为2维，shape为(G, H)。<br>
&emsp;&emsp;(3)无可选输入groupIdxOptional时：

$$
out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
$$

&emsp;&emsp;其中，gradY共3维，G, C, H依次表示gradY第0-2维的大小，计算后out为2维，shape为(G, H)。

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
      <td>gradY</td>
      <td>输入</td>
      <td>反向传播梯度，公式中的输入gradY。支持<a href="../../docs/zh/context/non_contiguous_tensor.md">非连续的Tensor</a>。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIdxOptional</td>
      <td>可选输入</td>
      <td>每个分组结束位置，公式中输入的groupIdxOptional。最多支持2048个组，支持非连续的Tensor。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>bias的梯度，公式中的out。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIdxType</td>
      <td>可选属性</td>
      <td>表示groupIdx的类型。支持的值为：<br>
      0：表示groupIdxOptional中的值为每个group的结束索引。<br>
      1：表示groupIdxOptional中的值为每个group的大小。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 当存在输入group_idx时，需要满足下列约束：
  - 需要确保张量的值不超过INT32的最大值并且是非负的。
  - grad_y仅支持2维形状。
- 当不存在输入group_idx时，grad_y仅支持3维形状。
- 当存在输入group_idx并且group_idx_type为0时，需要确保张量数据按升序排列，最后一个数值等于grad_y的第0维度的大小。
- 当存在输入group_idx并且group_idx_type为1时，必须确保张量数据的总和必须等于grad_y的第0维度的大小。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_bias_add_grad](./examples/test_aclnn_grouped_bias_add_grad.cpp) | 通过[aclnnGroupedBiasAddGrad](./docs/aclnnGroupedBiasAddGrad.md)接口方式调用GroupedBiasAddGrad算子。 |
| 图模式调用 | [test_geir_grouped_bias_add_grad](./examples/test_geir_grouped_bias_add_grad.cpp)   | 通过[算子IR](./op_graph/grouped_bias_add_grad_proto.h)构图方式调用GroupedBiasAddGrad算子。 |
