# aclnnBiasAdd

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：为输入张量 `x` 的每一个元素加上其对应通道（C 维）的偏差值 `bias`。
- 计算公式：

$$out_i = x_i + bias_{c(i)}$$

  其中 $c(i)$ 为元素 $i$ 在 `dataFormat` 指定的 C（通道）维上的下标。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnBiasAddGetWorkspaceSize` 接口获取入参并根据流程计算所需 workspace 大小，再调用 `aclnnBiasAdd` 接口执行计算。

- `aclnnStatus aclnnBiasAddGetWorkspaceSize(const aclTensor *x, const aclTensor *bias, char *dataFormatOptional, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnBiasAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBiasAddGetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
  | ------ | --------- | ---- | -------- | -------- |
  | x | 输入 | 待计算的输入张量 | FLOAT、FLOAT16、BFLOAT16、INT32 | NCHW、NHWC、NDHWC、NCDHW、ND |
  | bias | 输入 | 累加偏差，长度等于 `x` 在 `dataFormat` 指定 C 维的大小 | 与 x 一致 | ND |
  | dataFormatOptional | 输入 | 指定 C 维位置的数据排布，可选值 NCHW/NHWC/NDHWC/NCDHW，默认 "NHWC" | char* | - |
  | out | 输出 | 计算结果 | 与 x 一致 | 与 x 一致 |
  | workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | uint64_t* | - |
  | executor | 输出 | 返回 op 执行器 | aclOpExecutor** | - |

- **返回值：**

  返回 aclnnStatus 状态码。第一段接口完成入参校验，出现以下错误码时请检查：
  - 返回 161001（ACLNN_ERR_PARAM_NULLPTR）：x、bias、out 任一为空指针。
  - 返回 161002（ACLNN_ERR_PARAM_INVALID）：x、bias、out 的数据类型不一致或不在支持范围；bias 长度与 C 维不匹配。

## aclnnBiasAdd

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 数据类型 |
  | ------ | --------- | ---- | -------- |
  | workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 | void* |
  | workspaceSize | 输入 | workspace 大小，由第一段接口返回 | uint64_t |
  | executor | 输入 | op 执行器，由第一段接口返回 | aclOpExecutor* |
  | stream | 输入 | 执行任务的 stream | aclrtStream |

- **返回值：**

  返回 aclnnStatus 状态码。

## 约束说明

- 仅支持 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>。
- `bias` 的长度必须等于 `x` 在 `dataFormat` 指定 C 维上的大小。

## 调用示例

完整可运行样例见 [`examples/test_aclnn_bias_add.cpp`](../examples/test_aclnn_bias_add.cpp)，核心调用流程：

```cpp
// 1. 获取 workspace 大小（dataFormat 为 char*）
char dataFormat[] = "NHWC";
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnBiasAddGetWorkspaceSize(x, bias, dataFormat, out, &workspaceSize, &executor);

// 2. 申请 workspace（若 workspaceSize > 0）并执行
void *workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
aclnnBiasAdd(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```
