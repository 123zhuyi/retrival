# BatchToSpaceND

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

- 算子功能：将批次维度的数据重新排列到空间维度，并裁剪空间维度。

- 功能描述：
  该算子是SpaceToBatchND的逆操作。它首先将输入张量的批次维度按照指定的块形状（block_shape）重新排列到空间维度中，然后根据裁剪参数（crops）裁剪空间维度。具体来说，它将批次维度中的数据分散到空间维度中，从而增加空间维度的大小，同时减少批次维度的大小。

- 计算公式：
  设输入张量x为N维，形状为 $[x_0, x_1, \ldots, x_{N-1}]$，block_shape为M维1D张量 $[b_0, b_1, \ldots, b_{M-1}]$，crops为 $M \times 2$ 的2D张量 $[[c_{00}, c_{01}], [c_{10}, c_{11}], \ldots, [c_{M-1,0}, c_{M-1,1}]]$，满足 $1 \leq M < N \leq 8$。
  
  输出张量y形状为 $[y_0, y_1, \ldots, y_{N-1}]$，计算方式如下：
  
  $$
  y_i = \begin{cases}
  \frac{x_0}{\prod_{j=0}^{M-1} b_j}, & i = 0 \\
  x_i \times b_{i-1} - c_{i-1,0} - c_{i-1,1}, & 1 \leq i \leq M \\
  x_i, & M+1 \leq i \leq N-1
  \end{cases}
  $$
  
  其中，$x_0$ 必须能够被 $\prod_{j=0}^{M-1} b_j$ 整除。

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
      <td>-</td>
    </tr>
    <tr>
      <td>crops</td>
      <td>输入</td>
      <td>表示裁剪量，2D张量，形状为[M, 2]，指定每个空间维度从顶部和底部（或左侧和右侧）裁剪的元素数量</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，与输入x具有相同的数据类型。输出形状根据block_shape和crops进行计算</td>
      <td>与x一致</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持BOOL。

## 约束说明

- 输入张量x的维度N必须满足2 ≤ N ≤ 8。
- block_shape的维度M必须满足1 ≤ M < N。
- block_shape的长度必须等于crops的第一维度长度。
- crops的形状必须为 [M, 2]。
- 输入张量的第0 维（batch维度）必须能够被block_shape中所有元素的乘积整除。
- block_shape中的每个元素必须大于0。
- crops中的每个元素必须是非负整数。
- 对于每个空间维度i（i = 1, 2, ..., M），裁剪后的维度大小必须大于等于0，即：x.shape[i] × block_shape[i-1] - crops[i-1][0] - crops[i-1][1] ≥ 0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 图模式调用 | [test_geir_batch_to_space_nd](./examples/test_geir_batch_to_space_nd.cpp) | 通过[算子IR](./op_graph/batch_to_space_nd_proto.h)构图方式调用BatchToSpaceND算子。 |
