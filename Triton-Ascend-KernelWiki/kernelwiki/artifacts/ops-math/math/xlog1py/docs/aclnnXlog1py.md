# aclnnXlog1py

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/xlog1py)

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

- 接口功能：计算x * log(1 + y)，当x == 0时结果为0。

- 计算公式：

$$
z_i =
\begin{cases}
0,                                & x_i = 0 \\
x_i \cdot \log(1 + y_i),          & x_i \neq 0
\end{cases}
$$

- x与y支持broadcast，输出shape为广播后的最大值shape。
- y为NaN时输出y原值。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnXlog1pyGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnXlog1py"接口执行计算。

```Cpp
aclnnStatus aclnnXlog1pyGetWorkspaceSize(
  const aclTensor  *x,
  const aclTensor  *y,
  const aclTensor  *z,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnXlog1py(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  aclrtStream        stream)
```

## aclnnXlog1pyGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
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
      <td>x</td>
      <td>输入</td>
      <td>表示乘数因子，对应公式中x。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型需与y、z保持一致。</li><li>shape需与y满足broadcast关系。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>表示log1p的自变量，对应公式中y。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型需与x保持一致。</li><li>shape需与x满足broadcast关系。</li></ul></td>
      <td>数据类型与x保持一致。</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>表示计算结果，对应公式中z。</td>
      <td>shape需为x与y broadcast后的shape。</td>
      <td>数据类型与x保持一致。</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的tensor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x、y或z的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、y、z的数据类型不匹配。</td>
    </tr>
    <tr>
      <td>x、y或z的shape维度不在支持的范围之内（最大8维）。</td>
    </tr>
    <tr>
      <td>x、y或z使用了私有格式（private format）。</td>
    </tr>
  </tbody></table>

## aclnnXlog1py

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnXlog1pyGetWorkspaceSize获取。</td>
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
  - aclnnXlog1py默认确定性实现。
- 输入数据类型必须为FLOAT、FLOAT16或BFLOAT16。
- shape维度范围为0-8。
- 不支持私有格式（private format），仅支持ND格式。
- 输入x、y，输出z的数据类型必须一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_xlog1py.h"

#define CHECK_RET(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("[FAIL] " msg "\n"); \
            return -1; \
        } \
    } while (0)

#define LOG_PRINT(msg, ...) printf(msg "\n", ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto i : shape) size *= i;
    return size;
}

// Broadcast index: map flat index in output to flat index in input
static int64_t BroadcastIdx(int64_t flat, const std::vector<int64_t>& inShape,
                             const std::vector<int64_t>& outShape)
{
    int inRank = (int)inShape.size();
    int outRank = (int)outShape.size();
    int64_t outIdx = 0, outStride = 1;
    for (int d = 0; d < outRank; d++) {
        int dimIdx = outRank - 1 - d;
        int64_t dim = outShape[dimIdx];
        int64_t coord = (flat / outStride) % dim;
        int inDimIdx = dimIdx - (outRank - inRank);
        int64_t inDim = (inDimIdx >= 0) ? inShape[inDimIdx] : 1;
        int64_t inCoord = (inDim == 1) ? 0 : coord;
        int inStride = 1;
        for (int dd = inRank - 1; dd > inDimIdx; dd--) inStride *= inShape[dd];
        outIdx += inCoord * inStride;
        outStride *= dim;
    }
    return outIdx;
}

std::vector<float> ComputeGolden(
    const std::vector<float>& x, const std::vector<int64_t>& shapeX,
    const std::vector<float>& y, const std::vector<int64_t>& shapeY,
    const std::vector<int64_t>& outShape)
{
    int64_t n = GetShapeSize(outShape);
    std::vector<float> result(n);
    for (int64_t i = 0; i < n; i++) {
        int64_t ix = BroadcastIdx(i, shapeX, outShape);
        int64_t iy = BroadcastIdx(i, shapeY, outShape);
        float fx = x[ix], fy = y[iy];
        if (fx == 0.0f) {
            result[i] = 0.0f;
        } else {
            result[i] = fx * std::log1p(fy);
        }
    }
    return result;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");
    return 0;
}

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc failed");
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy H2D failed");

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int RunXlog1py(const std::vector<int64_t>& shapeX, const std::vector<float>& dataX,
               const std::vector<int64_t>& shapeY, const std::vector<float>& dataY,
               const std::string& tag, aclrtStream stream)
{
    LOG_PRINT("--- Test %s ---", tag.c_str());

    // Compute broadcast output shape
    int rank = std::max(shapeX.size(), shapeY.size());
    std::vector<int64_t> outShape(rank);
    for (int d = 0; d < rank; d++) {
        int dx = d - (rank - shapeX.size());
        int dy = d - (rank - shapeY.size());
        auto sx = (dx >= 0) ? shapeX[dx] : 1;
        auto sy = (dy >= 0) ? shapeY[dy] : 1;
        outShape[d] = std::max(sx, sy);
    }
    int64_t outSize = GetShapeSize(outShape);

    // Compute golden
    auto golden = ComputeGolden(dataX, shapeX, dataY, shapeY, outShape);

    // Allocate device tensors
    aclTensor* aclX = nullptr; void* devX = nullptr;
    auto ret = CreateAclTensor(dataX, shapeX, &devX, aclDataType::ACL_FLOAT, &aclX);
    CHECK_RET(ret == 0, "create tensor X failed");

    aclTensor* aclY = nullptr; void* devY = nullptr;
    ret = CreateAclTensor(dataY, shapeY, &devY, aclDataType::ACL_FLOAT, &aclY);
    CHECK_RET(ret == 0, "create tensor Y failed");

    std::vector<float> outHostData(outSize, 0);
    aclTensor* aclOut = nullptr; void* devOut = nullptr;
    ret = CreateAclTensor(outHostData, outShape, &devOut, aclDataType::ACL_FLOAT, &aclOut);
    CHECK_RET(ret == 0, "create tensor Out failed");

    // Phase 1: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnXlog1pyGetWorkspaceSize(aclX, aclY, aclOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnXlog1pyGetWorkspaceSize failed");

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "allocate workspace failed");
    }

    // Phase 2: Execute
    ret = aclnnXlog1py(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnXlog1py execute failed");

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed");

    // Copy result back and compare
    std::vector<float> npuResult(outSize, 0);
    ret = aclrtMemcpy(npuResult.data(), outSize * sizeof(float), devOut,
                      outSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy result D2H failed");

    bool pass = true;
    for (int64_t i = 0; i < outSize; i++) {
        float g = golden[i];
        float r = npuResult[i];
        double mere = (std::fabs(g) > 1e-6) ?
            std::fabs(r - g) / std::fabs(g) : std::fabs(r - g);
        if (mere > 0.001) {
            LOG_PRINT("  [FAIL][%s][%ld] golden=%.6f npu=%.6f", tag.c_str(), i, g, r);
            pass = false;
        }
    }
    if (pass) LOG_PRINT("  [PASS][%s] all %ld elems OK", tag.c_str(), outSize);

    // Cleanup
    aclDestroyTensor(aclX); aclDestroyTensor(aclY); aclDestroyTensor(aclOut);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devOut);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    return pass ? 0 : -1;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, "Init failed");

    int numPass = 0, numFail = 0;

    // Test 1: same shape [1,2,4,4]
    {
        std::vector<int64_t> shape = {1, 2, 4, 4};
        std::vector<float> x(32), y(32);
        for (int i = 0; i < 32; i++) { x[i] = 2.0f; y[i] = 1.0f; }
        if (RunXlog1py(shape, x, shape, y, "same_shape", stream) == 0) numPass++; else numFail++;
    }

    // Test 2: broadcast x=[1,2,1,4]  y=[1,2,4,4]
    {
        std::vector<int64_t> shapeX = {1, 2, 1, 4};
        std::vector<int64_t> shapeY = {1, 2, 4, 4};
        std::vector<float> x(8), y(32);
        for (int i = 0; i < 8; i++)  x[i] = 3.0f;
        for (int i = 0; i < 32; i++) y[i] = 2.0f;
        if (RunXlog1py(shapeX, x, shapeY, y, "broadcast", stream) == 0) numPass++; else numFail++;
    }

    // Test 3: x == 0 boundary case
    {
        std::vector<int64_t> shape = {1, 1, 8, 8};
        std::vector<float> x(64, 0.0f);
        std::vector<float> y(64, 100.0f);
        if (RunXlog1py(shape, x, shape, y, "x_eq_0", stream) == 0) numPass++; else numFail++;
    }

    LOG_PRINT("========================================");
    LOG_PRINT("ACLNN Xlog1py NPU results: PASS=%d  FAIL=%d", numPass, numFail);
    LOG_PRINT("========================================");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (numFail == 0) ? 0 : -1;
}
```
