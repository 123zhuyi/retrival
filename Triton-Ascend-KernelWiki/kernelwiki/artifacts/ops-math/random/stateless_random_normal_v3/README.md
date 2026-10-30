# StatelessRandomNormalV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：根据指定的均值（mean）和标准差（stdev）生成服从正态分布的随机数张量，使用基于计数器的Philox随机数生成算法，支持mean和stdev为标量或张量。

- 计算公式：

  $$
  result = StatelessRandomNormalV2 + Mul(std) + Add(mean)
  $$

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
      <td>shape</td>
      <td>输入</td>
      <td>输出张量的形状。</td>
      <td>INT64、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>用于基于计数器的随机数生成算法的秘钥。</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>输入</td>
      <td>用于基于计数器的随机数生成算法的初始计数值。</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>正态分布的均值，支持标量或与输出同形状的张量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stdev</td>
      <td>输入</td>
      <td>正态分布的标准差，支持标量或与输出同形状的张量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出指定形状的正态分布随机值。</td>
      <td>FLOAT、BF16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>可选属性，指定输出数据类型。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_normal_float_float](../stateless_random_normal_v2/examples/test_aclnn_normal_float_float.cpp) | 通过[aclnnNormalFloatFloat](../stateless_random_normal_v2/docs/aclnnNormalFloatFloat.md)接口方式调用stateless_random_normal_v3算子。 |
| aclnn调用 | [test_aclnn_normal_float_tensor](../stateless_random_normal_v2/examples/test_aclnn_normal_float_tensor.cpp) | 通过[aclnnNormalFloatTensor](../stateless_random_normal_v2/docs/aclnnNormalFloatTensor.md)接口方式调用stateless_random_normal_v3算子。 |
| aclnn调用 | [test_aclnn_normal_tensor_float](../stateless_random_normal_v2/examples/test_aclnn_normal_tensor_float.cpp) | 通过[aclnnNormalFloatTensor](../stateless_random_normal_v2/docs/aclnnNormalTensorFloat.md)接口方式调用stateless_random_normal_v3算子。 |
| aclnn调用 | [test_aclnn_normal_tensor_tensor](../stateless_random_normal_v2/examples/test_aclnn_normal_tensor_tensor.cpp) | 通过[aclnnNormalTensorTensor](../stateless_random_normal_v2/docs/aclnnNormalTensorTensor.md)接口方式调用stateless_random_normal_v3算子。 |
