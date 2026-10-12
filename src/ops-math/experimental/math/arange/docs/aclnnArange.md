# aclnnArange

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：从 `start` 起始、以 `step` 为步长、到 `end` 结束（左闭右开，不含 `end`），生成一个一维等差序列张量并写入 `out`。`start`、`end`、`step` 为 Host 侧标量（aclScalar），`out` 为一维输出张量（aclTensor）。功能与昇腾内置 `aclnnArange`、PyTorch `torch.arange` 一致，取整口径采用 `ceil`。

- 计算公式：

  序列元素：

  $$
  out_i = start + i \times step,\quad i = 0, 1, \dots, N-1
  $$

  输出元素个数 N（左闭右开，向上取整）：

  $$
  N = \left\lceil \frac{end - start}{step} \right\rceil
  $$

  其中 `out` 的元素个数 N 由调用方按上式计算并据此构造 `out` 张量（shape 为 `[N]`）。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnArangeGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnArange"接口执行计算。

```Cpp
aclnnStatus aclnnArangeGetWorkspaceSize(
  const aclScalar  *start,
  const aclScalar  *end,
  const aclScalar  *step,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnArange(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnArangeGetWorkspaceSize

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
      <td>start（aclScalar*）</td>
      <td>输入</td>
      <td>Host侧标量，序列起始值，对应公式中start。</td>
      <td><ul><li>step大于0时需满足start小于end；step小于0时需满足start大于end。</li><li>数据类型需与end、step、out一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end（aclScalar*）</td>
      <td>输入</td>
      <td>Host侧标量，序列结束值（左闭右开，不含end），对应公式中end。</td>
      <td><ul><li>取值约束同start。</li><li>数据类型需与start、step、out一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>step（aclScalar*）</td>
      <td>输入</td>
      <td>Host侧标量，步长，对应公式中step。</td>
      <td><ul><li>step不等于0。</li><li>数据类型需与start、end、out一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>一维输出张量，存放等差序列，对应公式中out。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为一维[N]，N=ceil((end-start)/step)，由调用方按该公式计算并构造。</li><li>数据类型需与start、end、step一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
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

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：start、end、step、out 支持 FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16，四者数据类型须保持一致。
  - UINT8 不可表示负值，UINT8 场景下 start、end、step 均需为非负且需满足 step 大于 0、start 小于 end。

- **返回值**

  aclnnStatus：返回状态码，具体参见 aclnn 返回码。

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
      <td>start、end、step、out 存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>start、end、step 或 out 的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>start、end、step、out 的数据类型不一致。</td>
    </tr>
    <tr>
      <td>step 等于 0，或 step 与 (end-start) 的符号关系不满足约束。</td>
    </tr>
  </tbody></table>

## aclnnArange

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnArangeGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见 aclnn 返回码。

## 约束说明

- 确定性说明：aclnnArange 默认确定性实现（纯逐元素等差序列生成，无 Reduce、无核间累加，相同输入恒产生相同输出）。
- start、end、step、out 四者的数据类型必须保持一致，不做跨数据类型推导。
- 需要满足 step 不等于 0；step 大于 0 时 start 小于 end，step 小于 0 时 start 大于 end。
- out 的元素个数 N 由调用方按 N=ceil((end-start)/step) 计算并构造 out 张量；算子不重新校验 N 与该公式的一致性。
- 整数类型（INT8、UINT8、INT16）输出当序列值超出对应类型值域时，按 CPU 标杆（numpy 对应 dtype）的取整与越界行为对齐。
- UINT8 不可表示负值，UINT8 场景下输入 start、end、step 均需为非负。

### 调用方前置约束（入参取值约束，由调用方保证，本接口不做取值校验）

> 本接口的入参校验仅覆盖**数据类型**（须属于支持的数据类型集合 FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16，且 start、end、step、out 四者一致，不满足返回 161002）与**空指针**（返回 161001）。
> 以下关于入参**取值**的约束属于**调用方前置条件**，接口**不做取值校验**（与"N 由调用方计算、算子不校验 N"的契约一致）。调用方须在调用前自行保证，否则行为未定义：

| 前置约束 | 调用方须保证 |
|---------|-------------|
| step ≠ 0 | step 非零 |
| step 符号匹配 | step>0 时 start<end；step<0 时 start>end（即 (end-start) 与 step 同号，N≥1） |
| UINT8 非负 | out.dtype==UINT8 时 start/end/step 均非负，且 step>0、start<end |
| N≥1 | N=ceil((end-start)/step) ≥ 1（不支持空 Tensor，N≤0 非法） |

## 调用示例

示例代码如下，仅供参考。完整示例见本算子 `../examples/test_aclnn_arange.cpp`，可通过 `bash build.sh --run_example arange eager cust --vendor_name=custom --experimental` 编译并运行。

示例展示 `aclnnArangeGetWorkspaceSize` + `aclnnArange` 两段式核心流程，覆盖 FLOAT 升序、FLOAT 负 step 降序、INT8 窄整型升序三组代表用例。**关键点**：`start`/`end`/`step` 用 `aclCreateScalar` 构造为 Host 侧标量；调用方按 `N = ceil((end-start)/step)` 计算元素个数并据此构造一维 `out` 张量（算子侧不计算、不校验 N）；其余 dtype（float16/bfloat16/uint8/int16）按相同模式替换 `aclDataType` 与标量/输出元素类型即可。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "aclnn_arange.h"

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

// 1) device / stream 初始化（固定写法）
int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 2) 调用方按 N = ceil((end - start) / step) 计算输出元素个数（算子侧不计算 / 不校验 N）
int64_t ComputeN(double start, double end, double step)
{
    return static_cast<int64_t>(std::ceil((end - start) / step));
}

// 3) 创建一维连续输出 aclTensor（仅分配 device 内存，无需拷入初值）
int CreateOutTensor(int64_t n, size_t elemSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = static_cast<size_t>(n) * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {n};       // 一维 [N]
    std::vector<int64_t> strides = {1};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, return -1);
    return 0;
}

// 4) 一组 aclnnArange 两段式调用（FLOAT 路径；其余浮点 dtype 改 aclDataType 与 elemSize 即可）
int RunFloatCase(const std::string& tag, aclrtStream stream, float start, float end, float step)
{
    int64_t n = ComputeN(start, end, step);          // 调用方算 N
    LOG_PRINT("---- %s (start=%g end=%g step=%g, N=%ld) ----\n", tag.c_str(), start, end, step, n);

    // 4.1 构造 start/end/step 三个 Host 侧标量（aclScalar），dtype 须四者一致
    aclScalar* sStart = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* sEnd = aclCreateScalar(&end, aclDataType::ACL_FLOAT);
    aclScalar* sStep = aclCreateScalar(&step, aclDataType::ACL_FLOAT);
    CHECK_RET(sStart && sEnd && sStep, LOG_PRINT("create scalar failed\n"); return -1);

    // 4.2 构造一维输出张量 out（shape=[N]，dtype 与标量一致）
    void* outDeviceAddr = nullptr;
    aclTensor* out = nullptr;
    auto ret = CreateOutTensor(n, sizeof(float), &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == 0, return ret);

    // 4.3 第一段：获取 workspace 大小与执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnArangeGetWorkspaceSize(sStart, sEnd, sStep, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArangeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4.4 按需申请 workspace（本算子 workspaceSize 通常为 0）
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 4.5 第二段：执行计算 + 同步等待
    ret = aclnnArange(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArange failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 4.6 拷回 host 并打印结果
    std::vector<float> result(n);
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(float), outDeviceAddr,
                      static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result D2H failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[%s] ->", tag.c_str());
    for (int64_t i = 0; i < n; i++) {
        LOG_PRINT(" %g", static_cast<double>(result[i]));
    }
    LOG_PRINT("\n");

    // 4.7 释放标量 / 张量 / device 内存
    aclDestroyScalar(sStart);
    aclDestroyScalar(sEnd);
    aclDestroyScalar(sStep);
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

// 5) 窄整型 INT8 路径（int16/uint8 同模式，仅改 aclDataType / 元素类型；uint8 须非负且 step>0）
int RunInt8Case(const std::string& tag, aclrtStream stream, int8_t start, int8_t end, int8_t step)
{
    int64_t n = ComputeN(start, end, step);
    LOG_PRINT("---- %s (start=%d end=%d step=%d, N=%ld) ----\n", tag.c_str(), start, end, step, n);

    aclScalar* sStart = aclCreateScalar(&start, aclDataType::ACL_INT8);
    aclScalar* sEnd = aclCreateScalar(&end, aclDataType::ACL_INT8);
    aclScalar* sStep = aclCreateScalar(&step, aclDataType::ACL_INT8);
    CHECK_RET(sStart && sEnd && sStep, LOG_PRINT("create scalar failed\n"); return -1);

    void* outDeviceAddr = nullptr;
    aclTensor* out = nullptr;
    auto ret = CreateOutTensor(n, sizeof(int8_t), &outDeviceAddr, aclDataType::ACL_INT8, &out);
    CHECK_RET(ret == 0, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnArangeGetWorkspaceSize(sStart, sEnd, sStep, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArangeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnArange(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArange failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<int8_t> result(n);
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(int8_t), outDeviceAddr,
                      static_cast<size_t>(n) * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result D2H failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[%s] ->", tag.c_str());
    for (int64_t i = 0; i < n; i++) {
        LOG_PRINT(" %d", static_cast<int>(result[i]));
    }
    LOG_PRINT("\n");

    aclDestroyScalar(sStart);
    aclDestroyScalar(sEnd);
    aclDestroyScalar(sStep);
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int main()
{
    // device / stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int failed = 0;
    // 用例 1：FLOAT 升序          start=0, end=10, step=1   -> [0,1,2,...,9]   (N=10)
    if (RunFloatCase("case1-FLOAT-asc", stream, 0.0f, 10.0f, 1.0f) != 0) { failed++; }
    // 用例 2：FLOAT 负 step 降序   start=5, end=-5, step=-2  -> [5,3,1,-1,-3]   (N=5)
    if (RunFloatCase("case2-FLOAT-neg-step", stream, 5.0f, -5.0f, -2.0f) != 0) { failed++; }
    // 用例 3：INT8 窄整型升序      start=-3, end=12, step=3  -> [-3,0,3,6,9]    (N=5)
    if (RunInt8Case("case3-INT8-asc", stream, -3, 12, 3) != 0) { failed++; }

    // 释放 device 资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failed == 0) {
        LOG_PRINT("ALL EXAMPLE CASES PASS\n");
        return 0;
    }
    LOG_PRINT("EXAMPLE FAILED: %d case(s) failed\n", failed);
    return 1;
}
```

> **说明**：
> - 其余 dtype 替换方式：float16 用 `aclDataType::ACL_FLOAT16`、bfloat16 用 `ACL_BF16`、uint8 用 `ACL_UINT8`、int16 用 `ACL_INT16`，并同步改对应标量/输出元素的 C++ 类型与 `sizeof`。
> - `start`/`end`/`step`/`out` 四者 dtype 必须一致；`N = ceil((end-start)/step)` 由调用方保证（详见上文「约束说明 → 调用方前置约束」），算子侧不重新校验。
> - UINT8 场景 `start`/`end`/`step` 均须非负且 `step>0`、`start<end`（uint8 不可表示负值）。
