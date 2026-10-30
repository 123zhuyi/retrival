# select

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

+ 算子功能：根据条件选取self或other中元素并返回（支持广播）。
+ 计算公式如下：

$$
out_i=where(self_i,other_i,condition_i)=\begin{cases}
  self_i, & \text{if condition}_i \\
  other_i, & \text{otherwise}
   \end{cases}
$$

## 函数原型

每个算子分为[两段式接口](../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSWhereGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSWhere”接口执行计算。

- `aclnnStatus aclnnSWhereGetWorkspaceSize(const aclTensor *condition, const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSWhere(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSWhereGetWorkspaceSize

* **参数说明**:
  - condition（aclTensor*，计算输入）：公式中的输入`condition`，Device侧的aclTensor，数据类型支持UINT8、BOOL，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，shape需要与self和other满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，数据维度不支持8维以上。

  - self（aclTensor*，计算输入）：公式中的输入`self`，数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/deduction_relationship.md)），Device侧的aclTensor，shape需要与other和condition满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、 INT32、 UINT64、 INT64、 UINT32、 FLOAT16、 UINT16、 INT16、 INT8、 UINT8、 DOUBLE、 BOOL、 COMPLEX64、 COMPLEX128、 BFLOAT16。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、 INT32、 UINT64、 INT64、 UINT32、 FLOAT16、 UINT16、 INT16、 INT8、 UINT8、 DOUBLE、 BOOL、 COMPLEX64、 COMPLEX128。

  - other（aclTensor*，计算输入）：公式中的输入`other`，数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/deduction_relationship.md)），Device侧的aclTensor，shape需要与self和condition满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、 INT32、 UINT64、 INT64、 UINT32、 FLOAT16、 UINT16、 INT16、 INT8、 UINT8、 DOUBLE、 BOOL、 COMPLEX64、 COMPLEX128、 BFLOAT16。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、 INT32、 UINT64、 INT64、 UINT32、 FLOAT16、 UINT16、 INT16、 INT8、 UINT8、 DOUBLE、 BOOL、 COMPLEX64、 COMPLEX128。

  - out（aclTensor \*，计算输出）：公式中的输出`out`，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，Device侧的aclTensor，shape需要是self与other和condition broadcast之后的shape。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、 INT32、 UINT64、 INT64、 UINT32、 FLOAT16、 UINT16、 INT16、 INT8、 UINT8、 DOUBLE、 BOOL、 COMPLEX64、 COMPLEX128、 BFLOAT16。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、 INT32、 UINT64、 INT64、 UINT32、 FLOAT16、 UINT16、 INT16、 INT8、 UINT8、 DOUBLE、 BOOL、 COMPLEX64、 COMPLEX128。

  - workspaceSize（uint64_t \*，出参）：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*，出参): 返回op执行器，包含了算子计算流程。
* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

```text
第一段接口完成入参校验，出现以下场景时报错：
返回161001 (ACLNN_ERR_PARAM_NULLPTR)：传入的self或other或condition或out是空指针。
返回161002 (ACLNN_ERR_PARAM_INVALID)：1.self或other或condition的数据类型和维度不在支持的范围之内。
                                      2.self和other无法做数据类型推导。
                                      3.self、other、condition broadcast推导失败或broadcast结果与out的shape不相同。
```

## aclnnSWhere

* **参数说明**:

  - workspace（void \*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnSWhereGetWorkspaceSize获取。
  - executor(aclOpExecutor \*，入参)：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。

* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnSWhere默认确定性实现。
