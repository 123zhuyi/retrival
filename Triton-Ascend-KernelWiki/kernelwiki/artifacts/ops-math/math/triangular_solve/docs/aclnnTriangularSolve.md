# aclnnTriangularSolve

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/triangular_solve)

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

- 接口功能：求解一个具有方形上或下三角形可逆矩阵A和多个右侧b的方程组。
- 计算公式：

  $$
  AX = b
  $$

  其中$A$是一个上三角方阵（当upper为false时为下三角方阵），其主对角线不含0的元素。$b$、$A$为二维矩阵或者二维矩阵的batch，当输入为batch时，返回输出的X也为对应的batch。当$A$的主对角线含有0，或元素非常接近0，且unitriangular为false时，输出结果可能包含$NaN$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnTriangularSolveGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnTriangularSolve`接口执行计算。

```Cpp
aclnnStatus aclnnTriangularSolveGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* A,
  bool             upper,
  bool             transpose,
  bool             unitriangular,
  aclTensor*       xOut,
  aclTensor*       mOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnTriangularSolve(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnTriangularSolveGetWorkspaceSize

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
      <td>公式中的b，方程右端项。</td>
      <td><ul><li>数据类型与A一致。</li><li>self[-2]=A[-2]。</li><li>除最后两个维度之外，A和self的其余维度满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>关系。</li></ul></td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>A（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的A，系数矩阵。</td>
      <td><ul><li>数据类型与self一致。</li><li>最后两个轴相等。</li><li>除最后两个维度之外，A和self的其余维度满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>关系。</li></ul></td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>upper（bool）</td>
      <td>输入</td>
      <td>控制公式中的A按上三角或下三角参与计算的计算属性。</td>
      <td>默认为true，A为上三角方阵，当upper为false时，A为下三角方阵。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose（bool）</td>
      <td>输入</td>
      <td>控制公式中使用A还是A<sup>T</sup>参与计算的计算属性。</td>
      <td>默认为false，当transpose为true时，计算A<sup>T</sup>X=b。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unitriangular（bool）</td>
      <td>输入</td>
      <td>控制公式中的A是否按单位三角矩阵处理的计算属性。</td>
      <td><ul><li>默认为false。</li><li>当unitriangular为true时，A的主对角线元素视为1，而不是从A引用。</li><li>当unitriangular为true时，输入self和A、输出xOut和mOut的数据类型只支持FLOAT。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的X，方程求解结果。</td>
      <td><ul><li>数据类型与self一致。</li><li>shape需要与<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>后的A、b满足AX=b约束。</li><li>A和self满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>关系之后的维度，最后一根轴dim=self[-1]。</li></ul></td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mOut（aclTensor*）</td>
      <td>输出</td>
      <td>broadcast后A的上三角（下三角）拷贝。</td>
      <td><ul><li>数据类型与self一致。</li><li>A和self满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>关系之后的维度，最后一根轴dim=A[-1]。</li></ul></td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>-</td>
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
      <td>传入的self、A、xOut、mOut中存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>self、A、xOut、mOut的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self、A、xOut、mOut的shape不符合约束。</td>
    </tr>
  </tbody></table>

## aclnnTriangularSolve

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
      <td>由第一段接口 <code>aclnnTriangularSolveGetWorkspaceSize</code> 获取的workspace大小。</td>
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

- 确定性说明：`aclnnTriangularSolve`默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_triangular_solve.h"

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

  // 2.构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {3, 1};
  std::vector<int64_t> otherShape = {3, 3};
  std::vector<int64_t> xOutShape = {3, 1};
  std::vector<int64_t> mOutShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* xOutDeviceAddr = nullptr;
  void* mOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* xOut = nullptr;
  aclTensor* mOut = nullptr;
  bool upper = true;
  bool transpose = false;
  bool unitriangular = false;
  std::vector<float> selfHostData = {1, 2, 3};
  std::vector<float> otherHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  std::vector<float> xOutHostData = {-0.2500, -0.1250, 0.5000};
  std::vector<float> mOutHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建xOut aclTensor
  ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT, &xOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mOut aclTensor
  ret = CreateAclTensor(mOutHostData, mOutShape, &mOutDeviceAddr, aclDataType::ACL_FLOAT, &mOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3.调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnTriangularSolve第一段接口
  ret = aclnnTriangularSolveGetWorkspaceSize(self, other, upper, transpose, unitriangular, xOut, mOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolveGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnTriangularSolve第二段接口
  ret = aclnnTriangularSolve(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolve failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto xSize = GetShapeSize(xOutShape);
  std::vector<float> resultData(xSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xOutDeviceAddr,
                    xSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  auto mSize = GetShapeSize(mOutShape);
  std::vector<float> mResultData(mSize, 0);
  ret = aclrtMemcpy(mResultData.data(), mResultData.size() * sizeof(mResultData[0]), mOutDeviceAddr,
                    mSize * sizeof(mResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < mSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, mResultData[i]);
  }

  // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(xOut);
  aclDestroyTensor(mOut);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(xOutDeviceAddr);
  aclrtFree(mOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
