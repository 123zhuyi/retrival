# ReduceSumOp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

算子功能：返回给定维度中输入张量每行的和。

## 参数说明

  - self（aclTensor*，计算输入）：Device侧的aclTensor，shape支持0-8维，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、BOOL、DOUBLE、COMPLEX64、COMPLEX128。输入为空tensor时，输出类型不能是复数类型COMPLEX64和COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。输入为空tensor时，输出类型不能是复数类型COMPLEX64和COMPLEX128。

  - dims（aclIntArray*，计算输入）：Host侧的aclIntArray，指定reduce维度，数据类型支持INT64，取值范围为[-self.dim(), self.dim()-1]。

  - keepDims（bool，计算输入）：Host侧的BOOL值，指定是否在输出张量中保留输入张量的维度。

  - dtype（aclDataType，计算输入）：Device侧的aclDataType，指定输出张量的数据类型。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT32、INT8、 INT16、 INT32、 INT64、UINT8、BOOL、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、FLOAT32、INT8、 INT16、 INT32、 INT64、UINT8、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

  - out（aclTensor*，计算输出）：Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、BOOL、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

## 约束说明

- 确定性计算：
  - 默认确定性实现。

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_reduce_sum](./examples/test_aclnn_reduce_sum.cpp) | 通过[aclnnReduceSum](docs/aclnnReduceSum.md)接口方式调用ReduceSum算子。 |
| 图模式调用  | [test_geir_reduce_sum.cpp](./examples/test_geir_reduce_sum.cpp) | 通过GE IR图模式调用ReduceSum算子。 |
