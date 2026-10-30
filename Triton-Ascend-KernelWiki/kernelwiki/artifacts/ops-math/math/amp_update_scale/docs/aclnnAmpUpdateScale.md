# aclnnAmpUpdateScale

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
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

- 接口功能：实现AMP（Automatic Mixed Precision）训练中的动态Scale更新逻辑。根据当前scale值、growth tracker计数器以及是否发现Inf/NaN，动态调整loss scale大小。

- 计算公式：

  $$
  \text{updated\_scale} = \begin{cases}
  \text{current\_scale} \times \text{backoff\_factor} & \text{if found\_inf} \neq 0 \\
  \text{current\_scale} \times \text{growth\_factor} & \text{if growth\_tracker + 1 = growth\_interval and new\_scale is finite} \\
  \text{current\_scale} & \text{otherwise}
  \end{cases}
  $$

  $$
  \text{updated\_growth\_tracker} = \begin{cases}
  0 & \text{if found\_inf} \neq 0 \text{ or growth triggered} \\
  \text{growth\_tracker} + 1 & \text{otherwise}
  \end{cases}
  $$

  说明：
  - 当found_inf不为0时，scale乘以backoff_factor回退，growth_tracker重置为0
  - 当found_inf为0且growth_tracker + 1等于growth_interval时，scale乘以growth_factor增长
  - 如果增长后的new_scale溢出（inf/nan），则保持当前scale不变，growth_tracker重置为0
  - 其他情况下，scale保持不变，growth_tracker递增1

- 使用场景：AMP训练中的动态损失缩放（Dynamic Loss Scaling），用于在FP16/BF16混合精度训练中防止梯度下溢。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnAmpUpdateScaleGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnAmpUpdateScale"接口执行计算。

```Cpp
aclnnStatus aclnnAmpUpdateScaleGetWorkspaceSize(
  const aclTensor* currentScale,
  const aclTensor* growthTracker,
  const aclTensor* foundInf,
  double           growthFactor,
  double           backoffFactor,
  int64_t          growthInterval,
  const aclTensor* updatedScale,
  const aclTensor* updatedGrowthTracker,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAmpUpdateScale(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnAmpUpdateScaleGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 160px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>currentScale（aclTensor*）</td>
      <td>输入</td>
      <td>当前的loss scale值。</td>
      <td>shape为标量 [1]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>growthTracker（aclTensor*）</td>
      <td>输入</td>
      <td>连续未出现Inf/NaN的步数计数器。</td>
      <td>shape为标量 [1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>foundInf（aclTensor*）</td>
      <td>输入</td>
      <td>是否检测到Inf/NaN的标志。</td>
      <td>shape为标量 [1]。0表示正常，非0表示发现Inf/NaN。数据类型需要与currentScale一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>growthFactor（float）</td>
      <td>输入</td>
      <td>scale增长因子。</td>
      <td>当连续growth_interval步未检测到Inf/NaN时，scale将乘以该因子。通常设置为2.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backoffFactor（float）</td>
      <td>输入</td>
      <td>scale回退因子。</td>
      <td>当检测到Inf/NaN时，scale将乘以该因子。通常设置为0.5。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>growthInterval（int64_t）</td>
      <td>输入</td>
      <td>触发scale增长的间隔步数。</td>
      <td>即连续多少步未检测到Inf/NaN后将增大scale。取值范围 >= 1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>updatedScale（aclTensor*）</td>
      <td>输出</td>
      <td>更新后的loss scale值。</td>
      <td>shape为 [1]。数据类型需要与currentScale的数据类型一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>updatedGrowthTracker（aclTensor*）</td>
      <td>输出</td>
      <td>更新后的growth tracker计数器。</td>
      <td>shape为 [1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md).

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>输入currentScale、growthTracker、foundInf的shape不是标量[1]。</td>
    </tr>
    <tr>
      <td>growthInterval超出取值范围[1, 2147483647]。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的currentScale、growthTracker、foundInf、updatedScale、updatedGrowthTracker是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>currentScale的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>foundInf的数据类型与currentScale不一致。</td>
    </tr>
    <tr>
      <td>updatedScale的数据类型与currentScale不一致。</td>
    </tr>
    <tr>
      <td>growthTracker的数据类型不是INT32。</td>
    </tr>
  </tbody>
  </table>

## aclnnAmpUpdateScale

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAmpUpdateScaleGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnAmpUpdateScale默认确定性实现。

- **数据类型约束**：current_scale与found_inf的数据类型必须一致；updated_scale的数据类型必须与current_scale一致。
- **shape约束**：所有输入输出张量均为标量，shape为 [1]。
- **growthInterval约束**：growthInterval取值范围为[1, 2147483647]。
- **Inf/NaN优先级**：found_inf不为0时，直接执行回退逻辑，忽略growth_tracker状态。
- **溢出保护**：当scale增长后的新值溢出（inf/nan）时，保持当前scale不变，growth_tracker重置为0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_amp_update_scale.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
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
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> scalarShape = {1};
  void* currentScaleDeviceAddr = nullptr;
  void* growthTrackerDeviceAddr = nullptr;
  void* foundInfDeviceAddr = nullptr;
  void* updatedScaleDeviceAddr = nullptr;
  void* updatedGrowthTrackerDeviceAddr = nullptr;
  aclTensor* currentScale = nullptr;
  aclTensor* growthTracker = nullptr;
  aclTensor* foundInf = nullptr;
  aclTensor* updatedScale = nullptr;
  aclTensor* updatedGrowthTracker = nullptr;

  // 创建currentScale
  std::vector<float> currentScaleHost = {65536.0f};
  ret = CreateAclTensor(currentScaleHost, scalarShape, &currentScaleDeviceAddr, ACL_FLOAT, &currentScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建growthTracker
  std::vector<int32_t> growthTrackerHost = {900};
  ret = CreateAclTensor(growthTrackerHost, scalarShape, &growthTrackerDeviceAddr, ACL_INT32, &growthTracker);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建foundInf
  std::vector<float> foundInfHost = {0.0f};
  ret = CreateAclTensor(foundInfHost, scalarShape, &foundInfDeviceAddr, ACL_FLOAT, &foundInf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输出updatedScale
  std::vector<float> updatedScaleHost = {0.0f};
  ret = CreateAclTensor(updatedScaleHost, scalarShape, &updatedScaleDeviceAddr, ACL_FLOAT, &updatedScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输出updatedGrowthTracker
  std::vector<int32_t> updatedGrowthTrackerHost = {0};
  ret = CreateAclTensor(updatedGrowthTrackerHost, scalarShape, &updatedGrowthTrackerDeviceAddr, ACL_INT32,
                        &updatedGrowthTracker);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用第一段接口，获取workspace大小和执行器
  float growthFactor = 2.0f;
  float backoffFactor = 0.5f;
  int64_t growthInterval = 1000;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnAmpUpdateScaleGetWorkspaceSize(currentScale, growthTracker, foundInf, growthFactor, backoffFactor,
                                            growthInterval, updatedScale, updatedGrowthTracker, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAmpUpdateScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 4. 根据workspaceSize申请workspace内存
  void* workspace = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 5. 调用第二段接口，执行计算
  ret = aclnnAmpUpdateScale(workspace, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAmpUpdateScale failed. ERROR: %d\n", ret); return ret);

  // 6. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 7. 将输出数据从device拷贝到host，并打印结果
  float updatedScaleVal = 0.0f;
  int32_t updatedGrowthTrackerVal = 0;
  ret = aclrtMemcpy(&updatedScaleVal, sizeof(float), updatedScaleDeviceAddr, sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy updatedScale failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(&updatedGrowthTrackerVal, sizeof(int32_t), updatedGrowthTrackerDeviceAddr, sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy updatedGrowthTracker failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("aclnnAmpUpdateScale result: updatedScale = %f, updatedGrowthTracker = %d\n", updatedScaleVal,
            updatedGrowthTrackerVal);

  // 8.（固定写法）释放资源
  aclDestroyTensor(currentScale);
  aclDestroyTensor(growthTracker);
  aclDestroyTensor(foundInf);
  aclDestroyTensor(updatedScale);
  aclDestroyTensor(updatedGrowthTracker);
  aclrtFree(currentScaleDeviceAddr);
  aclrtFree(growthTrackerDeviceAddr);
  aclrtFree(foundInfDeviceAddr);
  aclrtFree(updatedScaleDeviceAddr);
  aclrtFree(updatedGrowthTrackerDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspace);
  }
  aclrtDestroyStream(stream);
  auto aclRet = aclrtResetDevice(deviceId);
  CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("reset device failed. ERROR: %d\n", aclRet); return aclRet);
  aclRet = aclFinalize();
  CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("finalize acl failed. ERROR: %d\n", aclRet); return aclRet);
  return 0;
}
```
