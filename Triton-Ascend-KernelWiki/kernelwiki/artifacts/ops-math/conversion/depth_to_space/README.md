# DepthToSpace

## 产品支持情况

| 产品                                                         | 是否支持 |
| :---------------------------------------------------------- | :-------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：将深度（通道）维度的数据移动到空间维度（高度和宽度）。该算子通过对输入张量的深度（通道）维度进行重新排列，将其转换为空间维度（高度和宽度）。具体来说，它将输入张量的深度维度按照指定的块大小（block_size）进行划分，并将这些深度块重新排列到空间维度中，从而增加空间维度的大小，同时减少通道维度的深度。

- 计算公式：
  设输入张量形状为：
  - 当data_format="NHWC"时：`[batch, height, width, channels]`
  - 当data_format="NCHW"时：`[batch, channels, height, width]`

  设block_size为B，输出张量形状为：
  - 当data_format="NHWC"时：`[batch, height*B, width*B, channels/(B*B)]`
  - 当data_format="NCHW"时：`[batch, channels/(B*B), height*B, width*B]`

  其中，B*B必须能够整除通道维度。

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
      <td>表示输入张量，支持多种数据类型。数据格式必须与data_format属性指定的格式一致</td>
      <td>FLOAT、FLOAT16、INT8、INT16、INT32、UINT8、UINT16、UINT32、INT64、UINT64、BFLOAT16</td>
      <td>NHWC、NCHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，与输入x具有相同的数据类型。输出形状根据block_size、mode和data_format进行计算</td>
      <td>与x一致</td>
      <td>NHWC、NCHW</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性（必需）</td>
      <td>表示空间块的尺寸大小，必须是大于等于2的整数。通道维度必须能被block_size的平方整除</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性（必需）</td>
      <td>表示数据重排的模式，支持"DCR"和"CRD"两种模式</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性（可选）</td>
      <td>表示数据格式，指定输入和输出的维度排列顺序。支持"NHWC"和"NCHW"两种格式，默认为"NHWC"</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入和输出张量的维度必须是4D
- 输入和输出的数据格式必须相同，且仅支持NCHW和NHWC格式
- block_size必须是大于等于2的整数
- 通道维度必须能被block_size的平方整除
- mode属性必须是"DCR"或"CRD"
- data_format属性必须与输入张量的实际格式一致

## 调用说明

| 调用方式  | 样例代码   | 说明    |
| :-------- | :---------- | :------------------------ |
|  图模式  |[test_geir_depth_to_space](examples/test_geir_depth_to_space.cpp) | 通过[算子IR](./op_graph/depth_to_space_proto.h)构图方式调用DepthToSpace算子。 |
