# AddLora

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

  将输入x根据输入索引indices，分别和对应的weightA，weightB相乘，然后将结果累加到输入y上并输出。

- 计算公式：

  给定输入张量x，最后一维的长度为2d，函数AddLora进行以下计算：

  1. 将x根据indices中的索引进行重排，对应同一组权重的x排列在一起。
  
  2. 循环每个Lora分组，分别拿相应的x和weightA做矩阵乘：

     $$
     Z1 = x_{i} \cdot weightA[i, layerIdx, :, :]
     $$
  3. 得到的Z1继续和weightB做矩阵乘：

     $$
     Z2 = Z1 \cdot weightB[i, layerIdx, :, :] \times scale
     $$
  4. 最终把Z2输出累加到y上：

     $$
     \text{out} = y[:, yOffset: yOffset+ySliceSize] + Z2
     $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 855px"><colgroup>
  <col style="width: 164px">
  <col style="width: 138px">
  <col style="width: 300px">
  <col style="width: 133px">
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
      <td>y</td>
      <td>输入</td>
      <td>公式中的输入y。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightB</td>
      <td>输入</td>
      <td>公式中的输入weightB。</td>
      <td>FLOAT16</td>
      <td>ND、NZ</td>
    </tr>
     <tr>
      <td>indices</td>
      <td>输入</td>
      <td>公式中的输入indices。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightAOptional</td>
      <td>输入</td>
      <td>公式中的输入weightA。</td>
      <td>FLOAT16</td>
      <td>ND、NZ</td>
    </tr>
    <tr>
      <td>layerIdx</td>
      <td>属性</td>
      <td><ul><li>表示层数索引。</li><li>值需要小于weightB的第二个维度L。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>缩放系数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOffset</td>
      <td>属性</td>
      <td><ul><li>y更新时的偏移量。</li><li>值需要小于y的第二个维度H3。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ySliceSize</td>
      <td>属性</td>
      <td><ul><li>y更新时的范围。</li><li>值需要小于y的第二个维度H3。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：weightB和weightAOptional的数据格式支持ND。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_add_lora](./examples/test_aclnn_add_lora.cpp) | 通过[aclnnAddLora](./docs/aclnnAddLora.md)接口方式调用AddLora算子。    |
| 图模式调用 | [test_geir_add_lora](./examples/test_geir_add_lora.cpp)   | 通过[算子IR](./op_graph/add_lora_proto.h)构图方式调用AddLora算子。 |
