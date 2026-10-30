# aclnnBincount

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/experimental/math/bincount)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：统计一维非负整数张量中各取值的出现次数。指定 `weights` 时，按对应权重累加。

- 计算公式：

  未指定 `weights` 时：

  $$
  out[k] = \sum_i \mathbb{1}(self_i = k)
  $$

  指定 `weights` 时：

  $$
  out[k] = \sum_{i:self_i=k} weights_i
  $$

  输出长度为：

  $$
  L = \max(\max_i(self_i) + 1, minlength)
  $$

  当 `self` 为空Tensor时，输出长度为 `minlength`，输出元素均为0。输出Tensor由调用方预先申请。

## 函数原型

本接口为[两段式接口](../../../../docs/zh/context/two_phase_api.md)。必须先调用 `aclnnBincountGetWorkspaceSize` 获取计算所需的workspace大小和执行器，再调用 `aclnnBincount` 执行计算。

```cpp
aclnnStatus aclnnBincountGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *weights,
    int64_t          minlength,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnBincount(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnBincountGetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | self | 输入 | 待统计的整数Tensor。 | 元素必须为非负整数；支持空Tensor。 | INT8、INT16、INT32、INT64、UINT8 | ND | 1 | √ |
  | weights | 输入 | `self` 中每个元素对应的权重。 | 可选参数，可传入空指针；非空时shape必须与 `self` 相同。 | FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL | ND | 1 | √ |
  | minlength | 输入 | 输出Tensor的最小长度。 | 取值必须大于等于0。 | INT64 | - | - | - |
  | out | 输出 | 计数或权重累加结果。 | 由调用方按输出长度预先申请，元素下标对应 `self` 的取值。 | INT32、INT64、FLOAT、DOUBLE | ND | 1 | √ |
  | workspaceSize | 输出 | 返回需要在Device侧申请的workspace大小。 | 不允许为空指针。 | - | - | - | - |
  | executor | 输出 | 返回op执行器，包含算子计算流程。 | 不允许为空指针。 | - | - | - | - |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回码 | 错误码 | 描述 |
  | --- | --- | --- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的 `self`、`out`、`workspaceSize` 或 `executor` 是空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `self`、`weights` 或 `out` 的数据类型不在支持范围内。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `self`、`weights` 或 `out` 不是一维Tensor，或使用了不支持的私有格式。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `weights` 非空且shape与 `self` 不一致。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `minlength` 小于0。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | `out` 的长度超过 $2^{32}$。 |

## aclnnBincount

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 |
  | --- | --- | --- |
  | workspace | 输入 | 在Device侧申请的workspace内存地址。workspaceSize为0时可传入空指针。 |
  | workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口 `aclnnBincountGetWorkspaceSize` 获取。 |
  | executor | 输入 | op执行器，包含算子计算流程。 |
  | stream | 输入 | 指定执行任务的Stream。 |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `self` 仅支持非负整数。负数在Kernel运行期被检测到时，接口执行失败且不产出结果。
- 调用方必须按 $\max(\max(self)+1, minlength)$ 预先申请 `out`，确保 `self` 中的值均落在 `out` 的有效下标范围内。
- `weights` 在内部统一转换为FLOAT参与累加；使用高精度或大整数权重时，需要考虑FLOAT转换带来的精度影响。
- DOUBLE输出采用位拼接方式实现。输出较大且所需私有直方图超过可用UB空间时不支持，具体上限与硬件UB容量有关。

## 调用示例

完整示例请参见[test_aclnn_bincount.cpp](../examples/test_aclnn_bincount.cpp)。核心调用流程如下：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnBincountGetWorkspaceSize(
    self, weights, minlength, out, &workspaceSize, &executor);
if (ret != ACL_SUCCESS) {
    return ret;
}

void *workspace = nullptr;
if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
}

ret = aclnnBincount(workspace, workspaceSize, executor, stream);
if (ret != ACL_SUCCESS) {
    return ret;
}

ret = aclrtSynchronizeStream(stream);
if (workspace != nullptr) {
    aclrtFree(workspace);
}
return ret;
```

具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。
