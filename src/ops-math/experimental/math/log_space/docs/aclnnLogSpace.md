# aclnnLogSpace

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

> 整型输出（INT8/INT16/INT32/UINT8）仅 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 支持；<term>Ascend 950PR/Ascend 950DT</term> 维持原有的 FLOAT/FLOAT16/BFLOAT16 输出。

## 功能说明

- 接口功能：创建一个大小为 $steps$ 的一维张量，其值在 $base^{start}$ 到 $base^{end}$ 之间按对数尺度均匀间隔（含端点），以 $base$ 为底。相比原接口**新增 INT8/INT16/INT32/UINT8 输出**（整型按向零取整）。
- 计算公式：

$$ result = \left(base^{start},\ base^{\left(start + \frac{end-start}{steps-1}\right)},\ \ldots,\ base^{end}\right) $$

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnLogSpaceGetWorkspaceSize` 获取计算所需 workspace 大小及执行器，再调用 `aclnnLogSpace` 执行计算。

```cpp
aclnnStatus aclnnLogSpaceGetWorkspaceSize(
  const aclScalar*    start,
  const aclScalar*    end,
  int64_t             steps,
  double              base,
  const aclTensor*    result,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnLogSpace(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnLogSpaceGetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  | -- | -- | -- | -- | -- | -- | -- | -- |
  | start (aclScalar*) | 输入 | LogSpace 的起始指数 | 整型标量按数值读为 float | FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、INT16、INT32、UINT8 | ND | - | √ |
  | end (aclScalar*) | 输入 | LogSpace 的结束指数 | 整型标量按数值读为 float | FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、INT16、INT32、UINT8 | ND | - | √ |
  | steps (int64_t) | 输入 | 序列中的元素数量 | 取值 0 ≤ steps ≤ UINT32_MAX | int64_t | - | - | - |
  | base (double) | 输入 | 对数空间的底数 | 取值 base > 0 | double | - | - | - |
  | result (aclTensor*) | 输出 | 输出的对数间隔序列张量 | 整型按向零取整，溢出饱和；INT8/INT16/INT32/UINT8 仅 <term>Atlas A2/A3</term> 支持 | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、UINT8 | ND | 1（一维，长度=steps） | √ |
  | workspaceSize (uint64_t*) | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - | - | - | - |
  | executor (aclOpExecutor**) | 输出 | 返回 op 执行器，包含算子计算流程 | - | - | - | - | - |

- **返回值：** `aclnnStatus`，返回状态码。第一段接口完成入参校验，出现以下场景报错：

  | 返回码 | 错误码 | 描述 |
  | -- | -- | -- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的 start、end 或 result 是空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | start/end/result 数据类型不在支持范围（含 result 的 dtype 在当前芯片上不支持，如 <term>Ascend 950PR/Ascend 950DT</term> 传入整型）；或 steps < 0 / steps > UINT32_MAX；或 base ≤ 0；或 result 非一维 / result.shape[0] ≠ steps。 |

## aclnnLogSpace

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 |
  | -- | -- | -- |
  | workspace | 输入 | 在 Device 侧申请的 workspace 内存地址。 |
  | workspaceSize | 输入 | workspace 大小，由 aclnnLogSpaceGetWorkspaceSize 获取。 |
  | executor | 输入 | op 执行器，包含算子计算流程。 |
  | stream | 输入 | 指定执行任务的 Stream。 |

- **返回值：** `aclnnStatus`，返回状态码。

## 约束说明

- **确定性计算：** aclnnLogSpace 默认确定性实现。
- **整型输出：** 按 `base^x` 向零取整（CAST_TRUNC，对齐 torch `.to(int)`）；溢出按饱和处理，应保证 `base^x` 落在输出 dtype 范围内。UINT8 要求 `base^x ≥ 0`（base>0 时恒成立）。INT8/UINT8 因 c220 无 `float→1字节` 直接 Cast，内部经 half 中转（对 ≤255 整数精确）。
- **输出形状：** result 为一维张量，长度 = steps。

## 调用示例

示例代码如下（仅供参考），完整覆盖 7 种 dtype 见 `examples/test_aclnn_log_space.cpp`。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_log_space.h"

#define CHECK_RET(cond, expr) do { if (!(cond)) { expr; } } while (0)
#define LOG_PRINT(msg, ...) do { printf(msg, ##__VA_ARGS__); } while (0)

int main() {
  // 1. 初始化
  int32_t deviceId = 0;
  aclrtStream stream;
  CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
  CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
  CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

  // 2. 构造输入/输出：result 也可为 ACL_INT32 / ACL_INT8 等整型
  float startValue = 0.0f, endValue = 5.0f;
  int64_t steps = 6; double base = 2.0;          // -> [1, 2, 4, 8, 16, 32]
  aclScalar* start = aclCreateScalar(&startValue, aclDataType::ACL_FLOAT);
  aclScalar* end   = aclCreateScalar(&endValue,   aclDataType::ACL_FLOAT);

  std::vector<int64_t> shape = {steps};
  std::vector<int64_t> strides = {1};
  void* outDeviceAddr = nullptr;
  aclrtMalloc(&outDeviceAddr, steps * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  aclTensor* out = aclCreateTensor(shape.data(), 1, aclDataType::ACL_INT32, strides.data(), 0,
                                   aclFormat::ACL_FORMAT_ND, shape.data(), 1, outDeviceAddr);

  // 3. 两段式调用
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto ret = aclnnLogSpaceGetWorkspaceSize(start, end, steps, base, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  ret = aclnnLogSpace(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSpace failed %d\n", ret); return ret);
  aclrtSynchronizeStream(stream);

  // 4. 取回结果
  std::vector<int32_t> result(steps, 0);
  aclrtMemcpy(result.data(), result.size() * sizeof(int32_t), outDeviceAddr,
              steps * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  for (int64_t i = 0; i < steps; i++) LOG_PRINT("result[%ld] = %d\n", i, result[i]);

  // 5. 释放
  aclDestroyScalar(start); aclDestroyScalar(end); aclDestroyTensor(out);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
  return 0;
}
```
