# ArgMinWithValue

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

算子功能：返回self中指定维度的最小值及其索引位置。最小值保存到out中，最小值的索引保存到indices中。如果keepdim为false，则不保留对应的轴；如果为true，则保留指定轴的维度值为1。

## 函数原型

- 每个算子分为[两段式接口](../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMinDimGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMinDim”接口执行计算。

- `aclnnStatus aclnnMinDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepdim, aclTensor *out, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnMinDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMinDimGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*，计算输入)：待转换的目标张量，Device侧的aclTensor。数据类型与out一致，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。支持[1, 8]维。
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、INT64、BOOL
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、INT64、BOOL、BFLOAT16
     * <term>Ascend 950PR/Ascend 950DT</term>：FLOAT、FLOAT16、INT64、BOOL、BFLOAT16、INT32

  - dim(int64_t，计算输入)：指定的维度，数据类型为INT64，取值范围在[-self.dim(), self.dim())。

  - keepdim(bool，计算输入)：reduce轴的维度是否保留，数据类型为BOOL。

  - out(aclTensor*，计算输出)：Device侧的aclTensor，且数据类型和self一致。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。如果keepdim为false，则输出维度为self维度减1；如果keepdim为true，则输出维度等于self维度。
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、INT64、BOOL
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、INT64、BOOL、BFLOAT16
     * <term>Ascend 950PR/Ascend 950DT</term>：FLOAT、FLOAT16、INT64、BOOL、BFLOAT16、INT32

  - indices(aclTensor*，计算输出)：Device侧的aclTensor，数据类型支持INT32。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。

  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

```text
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、out和indices是空指针时。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. self、out、indices数据类型不在支持的范围内时。
                                    2. dim超出输入self的维度范围时。
```

## aclnnMinDim

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnMinDimGetWorkspaceSize获取。

  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMinDim默认确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_arg_min_with_value](./examples/test_aclnn_arg_min_with_value.cpp) | 通过[aclnnMinDim](./docs/aclnnMinDim.md)接口方式调用ArgMinWithValue算子。 |
