# aclnnQr

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/q_r)

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
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

- 接口功能：对输入Tensor进行正交分解。
- 计算公式：

  $$
  A = QR
  $$

  其中$A$为输入Tensor，维度至少为2，A可以表示为正交矩阵$Q$与上三角矩阵$R$的乘积的形式。

- 示例：

  ```text
  A = tensor([[1, 2], [3, 4]], dtype=torch.float)
  Q, R = QR(A, some=False)
  Q = tensor([[-0.3162, -0.9487],
              [-0.9487, 0.3162]])
  R = tensor([[-3.1623, -4.4272],
              [0.0000, -0.6325]])
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnQrGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnQr`接口执行计算。

```Cpp
aclnnStatus aclnnQrGetWorkspaceSize(
  const aclTensor* self,
  bool             some,
  aclTensor*       Q,
  aclTensor*       R,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnQr(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnQrGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的A。</td>
      <td>shape维度至少为2且不大于8，shape形如[..., M, N]，其中...表示0-6维。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>some（bool）</td>
      <td>输入</td>
      <td>控制公式中的Q、R输出形态的计算属性。</td>
      <td><ul><li>设为false时，Q为方阵，例如A[..., M, N]，输出完整的Q[..., M, M]、R[..., M, N]。</li><li>设为true时，Q为瘦矩阵，例如A[..., M, N]，输出Q[..., M, K]、R[..., K, N]，其中K为M、N的最小值。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Q（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的Q，正交分解输出的正交矩阵。</td>
      <td>shape约束参考<code>some</code>参数说明，且数据格式需要与<code>self</code>、R一致。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>由some推导</td>
      <td>√</td>
    </tr>
    <tr>
      <td>R（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的R，正交分解输出的上三角矩阵。</td>
      <td>shape约束参考<code>some</code>参数说明，且数据格式需要与<code>self</code>、Q一致。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>由some推导</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、Q、R中存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>self、Q、R的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self、Q、R的shape不符合约束。</td>
    </tr>
  </tbody></table>

## aclnnQr

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>由第一段接口 <code>aclnnQrGetWorkspaceSize</code> 获取的workspace大小。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：`aclnnQr`默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_qr.h"

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

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& qShape, std::vector<int64_t>& rShape, void** selfDeviceAddr,
    aclTensor** self, void** qDeviceAddr, aclTensor** q, void** rDeviceAddr, aclTensor** r)
{
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> qHostData = {0, 0, 0, 0};
  std::vector<float> rHostData = {0, 0, 0, 0};
  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建q,r aclTensor
  ret = CreateAclTensor(qHostData, qShape, qDeviceAddr, aclDataType::ACL_FLOAT, q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(rHostData, rShape, rDeviceAddr, aclDataType::ACL_FLOAT, r);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclTensor* q, aclTensor* r)
{
    aclDestroyTensor(self);
    aclDestroyTensor(q);
    aclDestroyTensor(r);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* qDeviceAddr, void* rDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
    aclrtFree(selfDeviceAddr);
    aclrtFree(qDeviceAddr);
    aclrtFree(rDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2.构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 2};
  bool some = false;
  std::vector<int64_t> qShape = {2, 2};
  std::vector<int64_t> rShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* qDeviceAddr = nullptr;
  void* rDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* q = nullptr;
  aclTensor* r = nullptr;

  ret = PrepareInputAndOutput(selfShape, qShape, rShape, &selfDeviceAddr, &self, &qDeviceAddr, &q, &rDeviceAddr, &r);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnQr第一段接口
  ret = aclnnQrGetWorkspaceSize(self, some, q, r, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnQr第二段接口
  ret = aclnnQr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQr failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(qShape);
  std::vector<float> resultQData(size, 0);
  ret = aclrtMemcpy(resultQData.data(), resultQData.size() * sizeof(resultQData[0]), qDeviceAddr,
                    size * sizeof(resultQData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result Q from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result Q[%ld] is: %f\n", i, resultQData[i]);
  }

  std::vector<float> resultRData(size, 0);
  ret = aclrtMemcpy(resultRData.data(), resultRData.size() * sizeof(resultRData[0]), rDeviceAddr,
                    size * sizeof(resultRData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result R from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result R[%ld] is: %f\n", i, resultRData[i]);
  }

  // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  ReleaseTensorAndScalar(self, q, r);

  // 7.释放device资源，需要根据具体API的接口定义参数
  ReleaseDevice(selfDeviceAddr, qDeviceAddr, rDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```
