# BitwiseAnd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算输入tensor中每个元素和输入标量的按位与结果。输入的tensor与other必须是整型或者BOOL型变量。当输入为BOOL时，计算逻辑与的结果。
- 计算公式：

  $$
  out_i = self_i \& other_i
  $$

## 参数说明

- self(aclTensor*，计算输入)：公式中的```self```，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，数据维度不支持8维以上。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT16、UINT16、INT32、INT64、INT8、UINT8、BOOL，且数据类型与other的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../docs/zh/context/deduction_relationship.md)），推导后的数据类型需在支持的数据类型范围内。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT16、UINT16、INT32、INT64、INT8、UINT8、BOOL，且数据类型需要与other满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/deduction_relationship.md)），推导后的数据类型需在支持的数据类型范围内。
- other(aclScalar*，计算输入)：公式中的```other```。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT16、UINT16、INT32、INT64、INT8、UINT8、BOOL，且数据类型与self的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../docs/zh/context/deduction_relationship.md)），推导后的数据类型需在支持的数据类型范围内。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT16、UINT16、INT32、INT64、INT8、UINT8、BOOL，且数据类型需要与self满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/deduction_relationship.md)），推导后的数据类型需在支持的数据类型范围内。
- out(aclTensor\*，计算输出)：公式中的```out```，数据类型需要是self与other推导之后可转换的数据类型（参见[互转换关系](../../docs/zh/context/conversion_relationship.md)），shape需要与self保持一致，[数据格式](../../docs/zh/context/data_format.md)支持ND，数据维度不支持8维以上。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8、FLOAT、FLOAT16、DOUBLE、BFLOAT16。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、UINT8、FLOAT、FLOAT16、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128。

## 约束说明

- 无

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_bitwise_and_scalar.cpp](examples/test_aclnn_bitwise_and_scalar.cpp) | 通过[aclnnBitwiseAndScalar](docs/aclnnBitwiseAndScalar.md)接口方式调用BitwiseAndScalar算子。 |
| aclnn接口  | [test_aclnn_bitwise_and_tensor.cpp](examples/test_aclnn_bitwise_and_tensor.cpp) | 通过[aclnnBitwiseAndTensor](docs/aclnnBitwiseAndTensor.md)接口方式调用BitwiseAndTensor算子。 |
