# Rsqrt

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：求input(Tensor)每个元素的平方根的倒数

- 计算公式：

$$
out = \frac{1}{\sqrt{input}}
$$

## 函数原型

- aclnnRsqrt和aclnnInplaceRsqrt实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnRsqrt：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceRsqrt：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnRsqrtGetWorkspaceSize”或者”aclnnInplaceRsqrtGetWorkspaceSize“接口获取入参并根据流程计算所需workspace大小，再调用“aclnnRsqrt”或者”aclnnInplaceRsqrt“接口执行计算。

  * `aclnnStatus aclnnRsqrtGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnRsqrt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceRsqrtGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)`
  * `aclnnStatus aclnnInplaceRsqrt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRsqrtGetWorkspaceSize

- **参数说明：**

  * self(aclTensor*，计算输入): 公式中的$input$，Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND,且shape需要与out一致。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、UINT8、INT8、INT16、INT32、INT64、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
  * out(aclTensor *，计算输出): 公式中的$out$，Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，且shape需要与self一致。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

```text
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1.传入的self或out是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. self和out的数据类型和数据格式不在支持的范围之内。
                                 2. self和out的shape不匹配
```

## aclnnRsqrt

- **参数说明：**

  * workspace(void *，入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnRsqrtGetWorkspaceSize获取。
  * executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## aclnnInplaceRsqrtGetWorkspaceSize

- **参数说明：**

  * selfRef(aclTensor *，计算输入): 公式中的$input$，Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，shape维度不大于8。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  * workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

```text
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1.传入的selfRef是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. selfRef的数据类型和数据格式不在支持的范围之内。
```

## aclnnInplaceRsqrt

- **参数说明：**

  * workspace(void *，入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceRsqrtGetWorkspaceSize获取。
  * executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnRsqrt&aclnnInplaceRsqrt默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/context/compile_and_run_sample.md)。
