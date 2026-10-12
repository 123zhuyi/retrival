# aclnnSlogdet

> 本文档对应 `experimental/math/slogdet` 的原生 AscendC 实现（registry-invoke 方式），接口与 `math/slogdet` 真值源保持一致。

## 产品支持情况

> 本次原生 AscendC 实现目标芯片为 Atlas A2/A3 训练/推理系列产品（Ascend 910B / 910C，其中 910C/A3 对应构建参数 `ascend910_93`），其余产品沿用 `math/slogdet` 的 aclnn 转发实现，不在本次交付范围内。

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：计算输入 `self` 的行列式的符号与行列式绝对值的自然对数，支持 batch 方阵输入，对标 `torch.linalg.slogdet`。
- 计算公式：

  $$
  signOut = sign(det(self))
  $$

  $$
  logOut = log(|det(self)|)
  $$

  其中 `det` 表示方阵行列式计算，`|·|` 表示绝对值（实数取绝对值）。如果 `det(self)` 的结果为 0，则 `logOut = -inf` 且 `signOut = 0`。

- 实现说明：内部经带部分主元的 LU 分解（`P·A = L·U`）求行列式，`logOut = Σ log|U_ii|`，`signOut` 由行置换奇偶性与各对角元符号合成。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnSlogdetGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnSlogdet` 接口执行计算。

```Cpp
aclnnStatus aclnnSlogdetGetWorkspaceSize(
  const aclTensor  *self,
  aclTensor        *signOut,
  aclTensor        *logOut,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnSlogdet(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSlogdetGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的<code>self</code>，输入方阵。</td>
      <td><ul><li>支持空Tensor。</li><li>shape满足(*, n, n)形式，其中<code>*</code>表示0或更多维度的batch，n为正整数，首版上界 1 ≤ n ≤ 4095（BLOCKED 路径 DataCopyPad blockCount uint16 限制，n>4095 返回错误，见「约束说明」）。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2及以上</td>
      <td>√</td>
    </tr>
    <tr>
      <td>signOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的<code>signOut</code>，行列式符号结果。</td>
      <td><ul><li>不支持空Tensor。</li><li>需要和<code>self</code>满足推导关系，数据类型与<code>self</code>一致。</li><li>shape与<code>self</code>的batch一致（self去掉最后两维）。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>与self的batch一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>logOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的<code>logOut</code>，行列式绝对值的自然对数结果。</td>
      <td><ul><li>不支持空Tensor。</li><li>需要和<code>self</code>满足推导关系，数据类型与<code>self</code>一致。</li><li>shape与<code>self</code>的batch一致（self去掉最后两维）。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>与self的batch一致</td>
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

  > 说明：本次原生 AscendC 实现首版仅支持 FLOAT。`math/slogdet` 转发实现额外支持 DOUBLE、COMPLEX64、COMPLEX128（self 为 COMPLEX 时 signOut/logOut 须为 COMPLEX），不在本实现范围内。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
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
      <td>传入的self、signOut、logOut中存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>self、signOut、logOut的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self的shape不满足约束（维度小于2或最后两维不相等，非方阵）。</td>
    </tr>
    <tr>
      <td>signOut和logOut的shape不满足约束（与self的batch不一致）。</td>
    </tr>
  </tbody></table>

## aclnnSlogdet

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnSlogdetGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：`aclnnSlogdet` 默认确定性实现。
- 数据类型限制：本次原生 AscendC 实现仅支持 FLOAT；self、signOut、logOut 数据类型需一致。
- 输入数据中不支持存在溢出值 `Inf` / `NaN`。
- 输入 `self` 维度必须大于等于 2，且最后两维必须相等（方阵）。
- **方阵维度 n 上界（首版 `1 ≤ n ≤ 4095`）**：BLOCKED（large-n）路径列 gather 用 `DataCopyPad` 的 `blockCount`（uint16，≤4095）；`n > 4095` 时该参数会静默越界，故 host 对 `n > 4095` 返回错误（不静默错误）。功能验证至 `n=512`，后续可分段 gather 放宽。
- 当 `det(self) = 0`（矩阵奇异）时，对应 batch 位置的 `logOut = -inf`、`signOut = 0`。

## 调用示例

示例代码如下，仅供参考。完整可运行版本见 `experimental/math/slogdet/examples/test_aclnn_slogdet.cpp`（真机 NPU 已 PASS），具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

> **链接注意（命中自定义 kernel 的关键）**：本算子的自定义 `libcust_opapi.so` 与上游 `libopapi.so` 都导出同名符号 `aclnnSlogdet` / `aclnnSlogdetGetWorkspaceSize`（上游为 `math/slogdet` 转发壳 → `LogMatrixDeterminant`）。编译示例时必须用 `-Wl,--no-as-needed ${CUSTOM_OP_LIBRARY}` 把自定义库置于链接命令最前（先于 `libopapi`），使其在 `DT_NEEDED` 中居首，运行时按 BFS 优先绑定到本算子的原生 AscendC kernel（`Slogdet`），否则会误绑上游 AICPU 转发实现。链接片段示例：
>
> ```cmake
> target_link_libraries(test_aclnn_slogdet
>     -Wl,--no-as-needed
>     ${CUSTOM_OP_LIBRARY}        # 自定义 libcust_opapi.so，必须居首
>     -Wl,--as-needed
>     ${ASCEND_PATH}/lib64/libascendcl.so
>     ${ASCEND_PATH}/lib64/libopapi.so
>     ${ASCEND_PATH}/lib64/libnnopbase.so)
> ```
>
> 运行前需保证自定义算子包已安装，并通过 `SLOGDET_CUSTOM_OPP` 指向 vendor 目录（如 `export SLOGDET_CUSTOM_OPP=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_math`），CMake 据此定位 `aclnn_slogdet_native.h` 与 `libcust_opapi.so`。完整工程（CMakeLists.txt / run.sh）见 `experimental/math/slogdet/examples/`。

```Cpp
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnn_slogdet_native.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 空 batch（标量输出 []）也分配 1 元素占位
    size_t allocSize = size == 0 ? sizeof(T) : size;
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (size > 0) {
        // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    //    self: [3, 2, 2] —— 3 个 2x2 方阵，覆盖正/负行列式与奇异（det=0）三种语义：
    //      [[2,0],[0,3]] → det=+6, sign=+1, log=ln6
    //      [[0,1],[1,0]] → det=-1, sign=-1, log=0（行交换置换）
    //      [[1,2],[2,4]] → det=0,  sign=0,  log=-inf（秩亏，演示退化语义）
    //    输出 signOut/logOut: [3]（self 去掉最后两维的 batch 形状）。
    std::vector<int64_t> selfShape = {3, 2, 2};
    std::vector<int64_t> outShape = {3};  // batch 形状
    int64_t batchCount = GetShapeSize(outShape);

    std::vector<float> selfHostData = {
        2, 0, 0, 3,  // det=+6
        0, 1, 1, 0,  // det=-1
        1, 2, 2, 4,  // det=0（奇异）
    };
    std::vector<float> signOutHostData(static_cast<size_t>(batchCount), 0.0f);
    std::vector<float> logOutHostData(static_cast<size_t>(batchCount), 0.0f);

    void* selfDeviceAddr = nullptr;
    void* signOutDeviceAddr = nullptr;
    void* logOutDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* signOut = nullptr;
    aclTensor* logOut = nullptr;

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(signOutHostData, outShape, &signOutDeviceAddr, aclDataType::ACL_FLOAT, &signOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(logOutHostData, outShape, &logOutDeviceAddr, aclDataType::ACL_FLOAT, &logOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnSlogdet 第一段接口（参数顺序：self, signOut, logOut），计算并获取
    //    workspace 大小与执行器 executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSlogdetGetWorkspaceSize(self, signOut, logOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 根据 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用 aclnnSlogdet 第二段接口执行计算
    ret = aclnnSlogdet(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdet failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 将 device 侧输出拷回 host 并打印（signOut ∈ {-1,0,+1}，奇异时 logOut=-inf）
    std::vector<float> signResult(static_cast<size_t>(batchCount), 0.0f);
    std::vector<float> logResult(static_cast<size_t>(batchCount), 0.0f);
    ret = aclrtMemcpy(
        signResult.data(), signResult.size() * sizeof(float), signOutDeviceAddr, batchCount * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy signOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(
        logResult.data(), logResult.size() * sizeof(float), logOutDeviceAddr, batchCount * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy logOut failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < batchCount; i++) {
        LOG_PRINT("  signOut[%ld] = %+.4f,  logOut[%ld] = %.6f\n", i, signResult[i], i, logResult[i]);
    }

    // 8. 释放资源
    aclDestroyTensor(self);
    aclDestroyTensor(signOut);
    aclDestroyTensor(logOut);
    aclrtFree(selfDeviceAddr);
    aclrtFree(signOutDeviceAddr);
    aclrtFree(logOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
