# TransformBiasRescaleQkv

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ | 
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：
  TransformBiasRescaleQkv算子是一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口。它用于调整这些向量的偏置（Bias）和缩放（Rescale）因子，以优化注意力计算过程。

- 计算公式：  
  逐个元素计算过程见公式：

  $$

  q_o=(q_i+q_{bias})/\sqrt{dim\_per\_head}\\

  $$
  $$
  
  k_o=k_i+k_{bias}\\

  $$
  $$
  
    v_o=v_i+v_{bias} 

  $$

  公式中：
  
  - dim_per_head为每个注意力头的维度。
  - q<sub>o</sub>、k<sub>o</sub>、v<sub>o</sub>分别为查询（Query）、键（Key）、值（Value）向量的输出元素。
  - q<sub>i</sub>、k<sub>i</sub>、v<sub>i</sub>分别为查询（Query）、键（Key）、值（Value）向量的输入元素。
  - q<sub>bias</sub>、k<sub>bias</sub>、v<sub>bias</sub>分别为查询（Query）、键（Key）、值（Value）向量的输入元素偏移。

## 参数说明

<table style="undefined;table-layout: fixed; width: 937px"><colgroup>
  <col style="width: 126px">
  <col style="width: 135px">
  <col style="width: 293px">
  <col style="width: 266px">
  <col style="width: 117px">
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
      <td>qkv</td>
      <td>输入</td>
      <td>公式中的输入q<sub>i</sub>、k<sub>i</sub>、v<sub>i</sub>。</td>
      <td>BFLOAT16、FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qkvBias</td>
      <td>输入</td>
      <td>公式中的输入q<sub>bias</sub>、k<sub>bias</sub>、v<sub>bias</sub>。</td>
      <td>BFLOAT16、FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numHeads</td>
      <td>属性</td>
      <td><ul><li>输入的头数。</li><li>取值大于0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qOut</td>
      <td>输出</td>
      <td>公式中的q<sub>o</sub>。</td>
      <td>BFLOAT16、FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kOut</td>
      <td>输出</td>
      <td>公式中的k<sub>o</sub>。</td>
      <td>BFLOAT16、FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vOut</td>
      <td>输出</td>
      <td>公式中的v<sub>o</sub>。</td>
      <td>BFLOAT16、FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

  - 输入qkv、qkvBias和输出qOut、kOut、vOut的数据类型需要保持一致。
  - 输入值为NaN，输出也为NaN，输入是Inf，输出也是Inf。
  - 输入是-Inf，输出也是-Inf。

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| aclnn调用  | [test_aclnn_transform_bias_rescale_qkv](./examples/test_aclnn_transform_bias_rescale_qkv.cpp) | 通过[aclnnTransformBiasRescaleQkv](./docs/aclnnTransformBiasRescaleQkv.md)接口方式调用TransformBiasRescaleQkv算子。   |
| 图模式调用 | -   | 通过[算子IR](./op_graph/transform_bias_rescale_qkv_proto.h)构图方式调用TransformBiasRescaleQkv算子。 |
