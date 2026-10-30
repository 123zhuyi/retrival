# Pow

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：exponent每个元素作为input对应元素的幂完成计算。

- 计算公式：

$$
out_i = input_i^{exponent_i}
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
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示底数。数据类型与exponent满足<a href ="../../docs/zh/context/deduction_relationship.md">TensorScalar互推导关系</a>。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT16、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exponent</td>
      <td>输入</td>
      <td>表示指数。数据类型与input满足<a href ="../../docs/zh/context/deduction_relationship.md">TensorScalar互推导关系</a>。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT16、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示input的exponent次幂。数据类型需要是input的数据类型与exponent的数据类型推导之后可转换的数据类型（参见<a href ="../../docs/zh/context/conversion_relationship.md">互转换关系</a>）。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT16、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

算子约束：INT32整型计算在如下范围以外的场景，会出现超时。

| shape  | exponent_value|
|----|----|
|<=100000（十万） |-200000000~200000000(两亿)|
|<=1000000（百万） |-20000000~20000000(两千万)|
|<=10000000（千万） |-2000000~2000000(两百万)|
|<=100000000（亿） |-200000~200000(二十万)|
|<=1000000000（十亿） |-20000~20000(两万)|

## 调用说明

| 调用方式 | 样例代码  | 说明     |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_exp2](./examples/test_aclnn_exp2.cpp) | 通过[aclnnExp2](./docs/aclnnExp2&aclnnInplaceExp2.md)接口方式调用Pow算子。 |
| aclnn调用 | [test_aclnn_pow_tensor_tensor](./examples/test_aclnn_pow_tensor_tensor.cpp) | 通过[aclnnPowTensorTensor](./docs/aclnnPowTensorTensor&aclnnInplacePowTensorTensor.md)接口方式调用Pow算子。 |
| 图模式调用 | [test\_geir\_pow](./examples/test_geir_pow.cpp) | 通过[算子IR](./op_graph/pow_proto.h)构图方式调用Pow算子。 |
