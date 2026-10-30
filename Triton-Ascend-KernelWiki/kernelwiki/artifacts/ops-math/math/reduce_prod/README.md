# ReduceProd

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

算子功能：返回输入tensor中所有元素的乘积。

## 参数说明

  - self(aclTensor*，计算输入)：Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL、COMPLEX64、COMPLEX128。输入为空tensor时，输出类型不能是复数类型COMPLEX64和COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL、COMPLEX64、COMPLEX128、BFLOAT16。输入为空tensor时，输出类型不能是复数类型COMPLEX64和COMPLEX128。

  - dtype(const aclDataType，计算输入)：Host侧的aclDataType，输出tensor的数据类型，需要与out的数据类型一致。

  - out(aclTensor*，计算输出)：Device侧的aclTensor，数据类型与dtype一致，若dtype未传入则数据类型与self一致。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。输出shape为(1)。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL、COMPLEX64、COMPLEX128、BFLOAT16。

## 约束说明

- 确定性计算：
  - 默认确定性实现。

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_prod](./examples/test_aclnn_prod.cpp) | 通过[aclnnProd](docs/aclnnProd.md)接口方式调用ReduceProd算子。 |
| aclnn接口 | [test_aclnn_prod_dim](./examples/test_aclnn_prod_dim.cpp) | 通过[aclnnProdDim](docs/aclnnProdDim.md)接口方式调用ReduceProd算子。 |
