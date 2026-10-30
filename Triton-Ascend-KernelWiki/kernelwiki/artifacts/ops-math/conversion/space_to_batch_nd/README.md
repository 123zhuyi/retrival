# SpaceToBatchND

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：将空间维度的数据按块重新排列到批次维度，并对空间维度补零。支持多维空间维度。

- 功能描述：
  该算子是BatchToSpaceND的逆操作。它首先根据paddings参数对输入的空间维度进行zero-padding，然后将padded空间划分为block_shape指定的块，每个块搬移到批次维度。输出批次维度变为输入的block_shape各元素乘积倍，空间维度相应缩小。

- 计算公式：
  设输入张量x为N维，形状为 $[x_0, x_1, \ldots, x_{N-1}]$，block_shape为M维1D张量 $[b_0, b_1, \ldots, b_{M-1}]$，paddings为 $M \times 2$ 的2D张量 $[[p_{00}, p_{01}], [p_{10}, p_{11}], \ldots, [p_{M-1,0}, p_{M-1,1}]]$，满足 $1 \leq M < N \leq 8$。
  
  输出张量y形状为 $[y_0, y_1, \ldots, y_{N-1}]$，计算方式如下：
  
  $$
  y_i = \begin{cases}
  x_0 \times \prod_{j=0}^{M-1} b_j, & i = 0 \\
  \frac{x_i + p_{i-1,0} + p_{i-1,1}}{b_{i-1}}, & 1 \leq i \leq M \\
  x_i, & M+1 \leq i \leq N-1
  \end{cases}
  $$
  
  其中，对于每个空间维度i，$x_i + p_{i-1,0} + p_{i-1,1}$ 必须能够被 $b_{i-1}$ 整除。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1480px">
  <colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
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
      <td>表示输入张量，支持多种数据类型</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BF16、FLOAT16、FLOAT、DOUBLE、BOOL、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_shape</td>
      <td>输入</td>
      <td>表示空间块的形状，1D张量，形状为[M]，指定每个空间维度的块大小</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>表示空间维度zero-padding量，2D张量，形状为[M, 2]，指定每个空间维度从头部和尾部补零的元素数量</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，与输入x具有相同的数据类型。输出形状根据block_shape和paddings进行计算</td>
      <td>与x一致</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持BOOL。

## 约束说明

1. 输入张量x的维度N必须满足2 ≤ N ≤ 8。
2. block_shape的维度M必须满足1 ≤ M < N。
3. block_shape的长度必须等于paddings的第一维度长度。
4. paddings的形状必须为 [M, 2]。
5. block_shape中的每个元素必须大于0。
6. paddings中的每个元素必须是非负整数。
7. 对于每个空间维度i（i = 1, 2, ..., M），padding后的维度大小必须能被block_shape[i-1]整除，即：(x.shape[i] + paddings[i-1][0] + paddings[i-1][1]) % block_shape[i-1] == 0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 图模式调用 | [test_geir_space_to_batch_nd](./examples/test_geir_space_to_batch_nd.cpp) | 通过[算子IR](./op_graph/space_to_batch_nd_proto.h)构图方式调用SpaceToBatchND算子。 |
