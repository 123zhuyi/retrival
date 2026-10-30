# ReduceLogSumExp

## 产品支持情况

| 产品                                                                | 是否支持 |
|:------------------------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                            |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                               |    ×     |
| <term>Atlas 推理系列产品 </term>                                        |    √     |
| <term>Atlas 训练系列产品</term>                                         |    √     |

## 功能说明

- 算子功能：返回输入tensor指定维度上的指数之和的对数。
- 计算公式：
  公式中i为dim指定的维度，j为输入在指定维度上的元素索引。
  $$
  logsumexp(x)_i = log\sum_{j} exp(x_{ij} )
  $$

- 示例

```text
例1：
  self: [2, 3, 4]       # self_shape=[2, 3, 4];
  dim: [2]              # 指定维度dim_shape=[2], dim_data = {1, 2};
  keepDim: false
  out: [2]              # out_shape=[2];

例2：
  self: [2, 3, 4]       # self_shape=[2, 3, 4];
  dim: [2]              # 指定维度dim_shape=[2], dim_data = {1, 2};
  keepDim: true
  out: [2, 1, 1]        # out_shape=[2, 1, 1];
```

## 参数说明

- self（aclTensor*，计算输入）：公式中的`self`，Device侧的aclTensor。shape支持0-8维，数据类型需要可转换成out数据类型（参见[互转换关系](../../docs/zh/context/conversion_relationship.md)）。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term> ：数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL。

- dim（aclIntArray*，计算输入）：参与计算的维度，公式中的`i`，Host侧的aclIntArray。取值范围为[-self.dim(), self.dim()-1]，支持的数据类型为INT64。

- keepDim（bool，计算输入）：决定reduce轴的维度是否保留，HOST侧的BOOL常量。

- out（aclTensor*，计算输入）：公式中的$logsumexp(x)$，Device侧的aclTensor。shape支持0-8维。若keepDim为true，除dim指定维度上的size为1以外，其余维度的shape需要与self保持一致；若keepDim为false，reduce轴的维度不保留，其余维度shape需要与self一致。数据类型需要可转换成self数据类型（参见[互转换关系](../../docs/zh/context/conversion_relationship.md)）。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term> ：数据类型支持FLOAT、FLOAT16、BFLOAT16。

- workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。

- executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_logsumexp.cpp](examples/test_aclnn_logsumexp.cpp) | 通过[aclnnLogSumExp](docs/aclnnLogSumExp.md)接口方式调用ReduceLogSumExp算子。 |
