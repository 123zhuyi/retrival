# BatchToSpace

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

- 算子功能：将批次维度的数据重新排列到空间维度，并裁剪空间维度。

- 功能描述：
  该算子是SpaceToBatch的逆操作。输入为4D NHWC张量`[N * bs * bs, H_in, W_in, C]`，输出为4D NHWC张量`[N, H_out, W_out, C]`。算子将输入batch维度中折叠的`bs × bs`个空间块还原到H和W维度，并通过crops参数裁剪空间边界。

- 计算公式：
  设输入x为4D NHWC张量`[N_in, H_in, W_in, C]`，block_size = bs，crops = `[[crop_top, crop_bottom], [crop_left, crop_right]]`。

  N_out = N_in / (bs × bs)，H_out = H_in × bs - crop_top - crop_bottom，W_out = W_in × bs - crop_left - crop_right。

  要求N_in能被bs × bs整除，且裁剪后的H_out、W_out > 0。

  输出y形状为`[N_out, H_out, W_out, C]`。

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
      <td>表示输入张量，4D NHWC张量 [N*bs*bs, H_in, W_in, C]，支持多种数据类型</td>
      <td>FLOAT、FLOAT16、INT8、INT16、INT32、UINT8、UINT16、INT64、DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>crops</td>
      <td>输入</td>
      <td>表示空间维度裁剪量，2D张量，形状为 [2, 2]，值为 [[crop_top, crop_bottom], [crop_left, crop_right]]</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性（必需）</td>
      <td>表示空间块的尺寸大小，必须是大于0的整数。N_in必须能被block_size × block_size整除</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，与输入x具有相同的数据类型。输出形状为 [N_out, H_out, W_out, C]</td>
      <td>与x一致</td>
      <td>NHWC</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入张量x必须为4D NHWC格式。
- block_size必须大于0。
- 输入batch维度N_in必须能被block_size × block_size整除。
- crops形状为 [2, 2]，每个元素 >= 0。
- 裁剪后空间维度必须大于0：H_in × block_size - crop_top - crop_bottom > 0，W_in × block_size - crop_left - crop_right > 0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  图模式  |[test_geir_batch_to_space](examples/test_geir_batch_to_space.cpp) | 通过[算子IR](./op_graph/batch_to_space_proto.h)构图方式调用BatchToSpace算子。 |
