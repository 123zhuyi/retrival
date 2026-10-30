# aclRfft1D

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。

### 计算公式

$$
{\displaystyle X_{k}=\sum _{n=0}^{N-1}x_{n}\cdot e^{-i2\pi {\tfrac {k}{N}}n}}
$$

### 示例

假设self为 {1, 2, 3, 4} 则out = {10, 0, -2, 2, -2, 0} = {10 + 0j, -2 + 2j, -2 + 0j}（自定义）

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclRfft1DGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclRfft1D”接口执行计算。

```Cpp
aclnnStatus aclRfft1DGetWorkspaceSize(
  const aclTensor*  self,
  int64_t           n,
  int64_t           dim,
  int64_t           norm,
  aclTensor*        out,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclRfft1D(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclRfft1DGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 150px">
    <col style="width: 200px">
    <col style="width: 350px">
    <col style="width: 80px">
    <col style="width: 80px">
    <col style="width: 80px">
    </colgroup>
      <thead>
        <tr>
          <th>参数名（类型）</th>
          <th>输入/输出</th>
          <th>描述</th>
          <th>使用说明</th>
          <th>数据类型</th>
          <th>格式</th>
          <th>shape</th>
          <th>非连续tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>self（aclTensor*）</td>
          <td>输入</td>
          <td>即公式中的输入。</td>
          <td>-</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>1-7维</td>
          <td>支持</td>
        </tr>
        <tr>
          <td>n（int64_t）</td>
          <td>输入</td>
          <td>表示信号长度。</td>
          <td>如果给定，则在计算Rfft1D之前，输入将被补零或修剪到此长度。n取值范围为[1, 4096]，2的n次幂最大值为262144。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>dim（int64_t）</td>
          <td>输入</td>
          <td>表示维度。</td>
          <td>如果给定，则RFFT将应用于指定的维度。支持的值为[-self.dim(), self.dim()-1]。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>norm（int64_t）</td>
          <td>输入</td>
          <td>表示归一化模式。</td>
          <td>支持取值：1表示不归一化，2表示按1/n归一化，3表示按1/sqrt(n)归一化。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>out（aclTensor*）</td>
          <td>输出</td>
          <td>表示公式中的输出。</td>
          <td>-</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>-</td>
          <td>支持</td>
        </tr>
        <tr>
          <td>workspaceSize（uint64_t*）</td>
          <td>输出</td>
          <td>返回需要在Device侧申请的workspace大小。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor（aclOpExecutor**）</td>
          <td>输出</td>
          <td>返回op执行器，包含了算子计算流程。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1150px">
    <colgroup>
    <col style="width: 380px">
    <col style="width: 135px">
    <col style="width: 635px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>输入或输出tensor为空。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>self的数据类型和维度不在支持的范围内。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>中间结果为null。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>执行者为null。</td>
      </tr>
    </tbody>
    </table>

## aclRfft1D

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 150px">
    <col style="width: 200px">
    <col style="width: 350px">
    <col style="width: 80px">
    <col style="width: 80px">
    <col style="width: 80px">
    </colgroup>
      <thead>
        <tr>
          <th>参数名（类型）</th>
          <th>输入/输出</th>
          <th>描述</th>
          <th>使用说明</th>
          <th>数据类型</th>
          <th>格式</th>
          <th>shape</th>
          <th>非连续tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>workspace（void*）</td>
          <td>输入</td>
          <td>在Device侧申请的workspace内存地址。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>workspaceSize（uint64_t）</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小。</td>
          <td>由第一段接口aclRfft1DGetWorkspaceSize获取。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor（aclOpExecutor*）</td>
          <td>输入</td>
          <td>op执行器，包含了算子计算流程。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>stream（aclrtStream）</td>
          <td>输入</td>
          <td>指定执行任务的Stream。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclRfft1D默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/acl_rfft1d.h"
#define CHECK_RET(cond, return_expr)   \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)
#define LOG_PRINT(message, ...)       \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
}
int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> selfShape = {1, 1, 8};
    std::vector<int64_t> outShape = {1, 1, 5, 2};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int n = 8;
    int dim = -1;
    int norm = 1; // backward - 1, forward - 2, ortho - 3

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclRfft1DGetWorkspaceSize(self, n, dim, norm, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1DGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclRfft1D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1D failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);

    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
