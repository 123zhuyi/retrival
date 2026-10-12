# aclnnDropoutV3Grad

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/random/drop_out_v3_grad)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：训练过程中，按照概率mask随机将输入中的元素置零，并将输出按照scale的比例放大。计算采用纯乘法链 `gradX = gradY * mask * scale`（mask为0或1的数值参与乘法）。
- 计算公式：

  $$
  gradX_i = gradY_i * mask_i * scale
  $$

  其中 $mask_i \in \{0, 1\}$ 为标量数值（由输入位掩码转换而来）。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnDropoutV3GradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDropoutV3Grad”接口执行计算。

```Cpp
aclnnStatus aclnnDropoutV3GradGetWorkspaceSize(
  const aclTensor* gradY,
  const aclTensor* mask,
  double           scale,
  aclTensor*       gradX,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnDropoutV3Grad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDropoutV3GradGetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 212px">
  <col style="width: 359px">
  <col style="width: 305px">
  <col style="width: 114px">
  <col style="width: 135px">
  <col style="width: 145px">
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
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>对应公式中的输入<code>gradY_i</code>。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mask（aclTensor*）</td>
      <td>输入</td>
      <td>位掩码数据，支持bit类型并使用UINT8类型存储。</td>
      <td><ul><li>支持空Tensor。</li><li>元素个数需要为(align(gradY的元素个数,128)/8)，表示比特数需要与128对齐，其中，align表示将gradY的元素个数向上对齐为128的倍数，如align(1111, 128)的结果为1152。</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale（double）</td>
      <td>输入</td>
      <td>对应公式中的输入<code>scale</code>，用于计算输出数据缩放因子。</td>
      <td>scale来自前向<code>1/(1-p)</code>，<code>p</code>为丢弃概率，取值<code>[0,1)</code>；因此scale取值为0或大于等于1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradX（aclTensor*）</td>
      <td>输出</td>
      <td>对应公式中的输出<code>gradX_i</code>。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型需要是gradY可转换的数据类型。</li><li>shape需要与gradY的一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradY、mask、gradX为空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>gradY、mask、gradX的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradY和gradX的shape不一致。</td>
    </tr>
    <tr>
      <td>mask的shape不满足条件。</td>
    </tr>
  </tbody>
  </table>

## aclnnDropoutV3Grad

- **参数说明：**

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 872px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDropoutV3GradGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnDropoutV3Grad默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dropout_v3_grad.h"

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
  // 1.（固定写法）device/stream初始化，参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // gradY: 反向梯度，shape {4, 2}，共 8 个元素
  // mask : 位掩码，元素个数为 align(8, 128)/8 = 16 (uint8)
  std::vector<int64_t> gradYShape = {4, 2};
  std::vector<int64_t> maskShape = {16};
  std::vector<int64_t> gradXShape = {4, 2};
  void* gradYDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  void* gradXDeviceAddr = nullptr;
  aclTensor* gradY = nullptr;
  aclTensor* mask = nullptr;
  aclTensor* gradX = nullptr;
  std::vector<float> gradYHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint8_t> maskHostData(16, 0);
  // mask 首字节 42 = 0b00101010，LSB-first展开后bit序为 0,1,0,1,0,1,0,0
  // 配合 scale=2，结果为 gradY_i * mask_i * scale：即 result = {0, 2, 0, 6, 0, 10, 0, 0}
  maskHostData[0] = 42;
  std::vector<float> gradXHostData(8, 0);
  // scale = 1 / (1 - p)，此处 p = 0.5 -> scale = 2
  double scale = 2;

  // 创建gradY aclTensor
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mask aclTensor
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_UINT8, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradX aclTensor
  ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用aclnnDropoutV3Grad执行dropout
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnDropoutV3Grad第一段接口
  ret = aclnnDropoutV3GradGetWorkspaceSize(gradY, mask, scale, gradX, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDropoutV3GradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnDropoutV3Grad第二段接口
  ret = aclnnDropoutV3Grad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDropoutV3Grad failed. ERROR: %d\n", ret); return ret);
  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradXShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(gradY);
  aclDestroyTensor(gradX);
  aclDestroyTensor(mask);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradYDeviceAddr);
  aclrtFree(maskDeviceAddr);
  aclrtFree(gradXDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
