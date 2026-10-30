# SpaceToBatch

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             | √        |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     | √        |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √        |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×     |
| <term>Atlas 推理系列产品</term>                             | √        |
| <term>Atlas 训练系列产品</term>                              | √        |

## 功能说明

- 算子功能：将空间维度的数据按块重新排列到批次维度，并对空间维度补零。

- 功能描述：
  该算子首先根据paddings参数对输入的空间维度进行zero-padding，然后将padded空间划分为`block_size × block_size`的块，每个块搬移到批次维度。输出批次维度变为输入的`block_size × block_size`倍，空间维度相应缩小。

- 计算公式：
  设输入x为4D NHWC张量`[N, H_in, W_in, C]`，block_size = bs，paddings = `[[pad_top, pad_bottom], [pad_left, pad_right]]`。

  H_padded = H_in + pad_top + pad_bottom，W_padded = W_in + pad_left + pad_right，

  H_out = H_padded / bs，W_out = W_padded / bs。

  要求H_padded和W_padded均能被bs整除。

  输出y形状为`[N * bs * bs, H_out, W_out, C]`。

  坐标映射（输出 → 输入）：

  ```text
  n     = n_out / (bs * bs)
  bh    = (n_out % (bs * bs)) / bs
  bw    = n_out % bs
  h_in  = h_out * bs + bh - pad_top
  w_in  = w_out * bs + bw - pad_left

  若0 ≤ h_in < H_in且0 ≤ w_in < W_in：
    y[n_out, h_out, w_out, c] = x[n, h_in, w_in, c]
  否则：
    y[n_out, h_out, w_out, c] = 0
  ```

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
      <td>表示输入张量，4D NHWC张量 [N, H_in, W_in, C]，支持多种数据类型</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>表示空间维度zero-padding量，2D张量，形状为 [2, 2]，值为 [[pad_top, pad_bottom], [pad_left, pad_right]]</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性（必需）</td>
      <td>表示空间块的尺寸大小，必须是大于0的整数。H_padded和W_padded必须能被block_size整除</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，与输入x具有相同的数据类型。输出形状为 [N*block_size*block_size, H_out, W_out, C]</td>
      <td>与x一致</td>
      <td>NHWC</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入张量x必须为4D NHWC格式。
- block_size必须大于0。
- paddings形状为 [2, 2]，每个元素 >= 0。
- H_padded = H_in + pad_top + pad_bottom必须能被block_size整除。
- W_padded = W_in + pad_left + pad_right必须能被block_size整除。
- block_size为编译期常量（算子属性），非运行时tensor输入。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  图模式  |[test_geir_space_to_batch](examples/test_geir_space_to_batch.cpp) | 通过[算子IR](./op_graph/space_to_batch_proto.h)构图方式调用SpaceToBatch算子。 |
