# MaskedScale

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：完成elementwise计算
- **计算公式**：

  $$
  out = self \times mask \times scale
  $$

## 函数原型

每个算子分为[两段式接口](../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMaskedScaleGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnMaskedScale”接口执行计算。

* `aclnnStatus aclnnMaskedScaleGetWorkspaceSize(const aclTensor* self, const aclTensor* mask, float scale, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMaskedScale(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMaskedScaleGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*，计算输入)：公式中的输入`self`，Device侧的aclTensor。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT，[数据格式](../../docs/zh/context/data_format.md)支持ND。

  - mask(aclTensor*，计算输入)：公式中的`mask`，Device侧的aclTensor，shape需要与self一致。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT8、INT8、FLOAT16、FLOAT，[数据格式](../../docs/zh/context/data_format.md)支持ND。

  - scale(float，计算输入)：进行数据缩放，数据类型支持FLOAT。

  - y(aclTensor\*，计算输出)：公式中的`out`，Device侧的aclTensor，数据类型和shape需要与self一致。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT，[数据格式](../../docs/zh/context/data_format.md)支持ND。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

  ```text
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1.传入的self、mask或y是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1.输入和输出的数据类型不在支持的范围之内。
                                    2.输出y和输入self数据类型不一致。
                                    3.self、mask和y的shape不一致。
  ```

## aclnnMaskedScale

- **参数说明：**

  * workspace(void\*，入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnMaskedScaleGetWorkspaceSize获取。
  * executor(aclOpExecutor\*，入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMaskedScale默认确定性实现。
