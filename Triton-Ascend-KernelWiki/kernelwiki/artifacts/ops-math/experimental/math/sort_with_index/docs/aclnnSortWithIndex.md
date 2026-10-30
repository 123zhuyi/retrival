# aclnnSortWithIndex

> 本文档为 `experimental/math/sort_with_index`（ascend910b 原生实现）的 aclnn 接口参考。
> L0 语义与接口真值源：`math/sort_with_index/`。真值源仅提供 L0 内部接口 `l0op::SortWithIndex`，**未提供** L2 公开 `aclnnSortWithIndex`；本 `experimental` 实现新增了 L2 公开两段式接口 `aclnnSortWithIndexGetWorkspaceSize` / `aclnnSortWithIndex`，并按 910B 首版交付（4 组 int32-index）定型，接口签名/约束/dtype 支持均与最终实现一致。

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 说明：真值源 `math/sort_with_index` 仅适配 <term>Ascend 950PR/Ascend 950DT</term>（arch35 kernel）。本 `experimental` 实现新增 <term>Atlas A2 训练系列产品（Ascend 910B）</term> 原生支持。本文档"是否支持"按 910B 实现交付情况填写，最终以 `op_host/sort_with_index_def.cpp` 的 `AddConfig` 与 kernel 实际能力为准。

## 功能说明

- **接口功能**：沿指定轴 `axis` 对输入张量 `x` 进行排序，输出排序后的数值张量 `y`，并将输入索引张量 `index` 按与 `x` 相同的排序顺序同步重排，输出 `sortedIndex`。等价于 PyTorch `torch.sort(x, dim=axis, descending=descending, stable=stable)` 语义，但索引来源为外部传入的 `index`（而非内部生成的 `0..N-1`）。
- **计算逻辑**：设排序轴上一维切片长度为 $N$，排序得到置换 $p$（$0..N-1$ 的一个排列），满足：

  升序（descending=false）：

  $$
  x[p_0] \le x[p_1] \le \cdots \le x[p_{N-1}]
  $$

  降序（descending=true）：

  $$
  x[p_0] \ge x[p_1] \ge \cdots \ge x[p_{N-1}]
  $$

  输出：

  $$
  y_k = x[p_k], \quad sortedIndex_k = index[p_k]
  $$

- **稳定性**：`stable=true` 时相等元素保持原相对顺序（ties 中原始 index 较小者在前）；`stable=false` 时 ties 顺序未定义。910B 实现的 Sort 路径天然稳定，`stable=true`/`false` 走相同内核路径。
- **特殊值**（910B 实现，D3=3b-B / I1 定型）：$+\infty > $ 有限数 $ > -\infty$；**NaN 升序落序列开头、降序落序列开头**（注意升序与 torch「NaN 排末尾」约定不同，如需 torch 兼容请调用前过滤 NaN）；NaN 行按 `isnan` 比较（视为 NaN，位型可能与输入不同，值语义不变）。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSortWithIndexGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnSortWithIndex"接口执行计算。

```cpp
aclnnStatus aclnnSortWithIndexGetWorkspaceSize(
  const aclTensor  *x,
  const aclTensor  *index,
  int64_t           axis,
  bool              descending,
  bool              stable,
  aclTensor        *valuesOut,
  aclTensor        *indicesOut,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnSortWithIndex(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSortWithIndexGetWorkspaceSize

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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>待排序的数值张量，对应公式中x。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需与index一致。</li><li>INT32须满足 <code>|x| ≤ 2^24</code>（经浮点排序的值域限制）。</li></ul></td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>index（aclTensor*）</td>
      <td>输入</td>
      <td>待跟随排序的原始索引张量，对应公式中index。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需与x一致。</li><li>910B 首版仅支持 INT32（INT64 暂不支持）。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>axis（int64_t）</td>
      <td>输入</td>
      <td>排序所沿的轴，对应公式中axis。默认-1。</td>
      <td>当前仅支持沿最后一维排序，即取值为-1或rank-1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending（bool）</td>
      <td>输入</td>
      <td>排序顺序，true为降序，false为升序。默认false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable（bool）</td>
      <td>输入</td>
      <td>是否稳定排序，true为稳定，false为非稳定。默认false。</td>
      <td>stable为true时，相等元素保持原相对顺序。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valuesOut（aclTensor*）</td>
      <td>输出</td>
      <td>排序后的数值张量，对应公式中y。</td>
      <td><ul><li>数据类型与x一致。</li><li>shape与x一致。</li></ul></td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>indicesOut（aclTensor*）</td>
      <td>输出</td>
      <td>跟随排序后的索引张量，对应公式中sortedIndex。</td>
      <td><ul><li>数据类型与index一致（910B 首版为 INT32）。</li><li>shape与index一致。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>0-8</td>
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
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：910B 原生实现首版定型支持 **4 组 dtype 组合**：value ∈ {FLOAT16、FLOAT、BFLOAT16、INT32} × index = INT32（sorted_index 跟随 index = INT32）。其中 INT32 value 须满足 `|x| ≤ 2^24`（经浮点排序键的值域限制）。INT64-index 及 INT8/INT16/UINT* 等其余 dtype 首版不支持（以实际交付的 `_def.cpp` 与 kernel 能力为准）。

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
      <td>x、index、y、sortedIndex存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x或index的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x与index的shape不一致。</td>
    </tr>
    <tr>
      <td>x的维度数不在[0, 8]范围之内。</td>
    </tr>
    <tr>
      <td>axis不等于-1且不等于rank-1（当前仅支持最后一维排序）。</td>
    </tr>
  </tbody></table>

## aclnnSortWithIndex

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnSortWithIndexGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnSortWithIndex默认确定性实现。
- x与index的shape必须一致；y、sortedIndex的shape分别与x、index一致。
- x的维度数需在[0, 8]范围内。
- axis当前仅支持沿最后一维排序（取值为-1或rank-1）。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 首版支持 4 组 dtype：value ∈ {FLOAT16、FLOAT、BFLOAT16、INT32} × index = INT32。
  - INT32 value 输入须满足 `|x| ≤ 2^24`（经浮点排序键的值域限制），否则排序结果可能不正确。
  - INT64-index 及 UINT*、INT8/INT16 等其余 dtype 当前不支持。
  - 排序轴长 N 存在 dtype 相关上限（约 FLOAT16 ~3008 / FLOAT32 ~2816 / BFLOAT16 ~2816 / INT32 ~2560，随实际可用 UB 运行时取值），超出上限将返回 tiling 失败（优雅拒绝，非崩溃）。
  - NaN 升序落序列开头、降序落序列开头（升序与 torch「NaN 排末尾」约定不同，如需 torch 兼容请调用前过滤 NaN）；NaN 行按 `isnan` 比较，位型可能与输入不同但值语义不变。$+\infty$/$-\infty$ 正常参与排序。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

> 完整可编译示例见 `experimental/math/sort_with_index/examples/test_aclnn_sort_with_index.cpp`（含编译运行脚本 `examples/run.sh`，已固化独立 vendor + `ASCEND_CUSTOM_OPP_PATH` 以规避系统 built-in `SortWithIndex`）。
>
> 本例演示 `float32` value + `int32` index 沿最后一维「升序」排序 + index 跟随重排：
> 输入 `x = [3.0, 1.0, 4.0, 1.5, 2.0]`、`index = [0, 1, 2, 3, 4]`，
> 升序输出 `y = [1.0, 1.5, 2.0, 3.0, 4.0]`、`sortedIndex = [1, 3, 4, 0, 2]`。

```Cpp
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sort_with_index.h"

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

static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
static int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                           aclDataType dataType, aclTensor **tensor)
{
    auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // ---- 0. 初始化 ACL ----
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return -1);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return -1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return -1);

    // ---- 1. 构造输入（float32 value + int32 index），沿最后一维升序 ----
    std::vector<int64_t> shape = {5};
    std::vector<float> xHost = {3.0f, 1.0f, 4.0f, 1.5f, 2.0f};
    std::vector<int32_t> indexHost = {0, 1, 2, 3, 4};
    int64_t axis = -1;        // 仅支持最后一维
    bool descending = false;  // 升序
    bool stable = false;

    int64_t n = GetShapeSize(shape);
    std::vector<float> yHost(static_cast<size_t>(n), 0.0f);
    std::vector<int32_t> sortedIndexHost(static_cast<size_t>(n), 0);

    void *xDev = nullptr;
    void *indexDev = nullptr;
    void *yDev = nullptr;
    void *sortedIndexDev = nullptr;
    aclTensor *x = nullptr;
    aclTensor *index = nullptr;
    aclTensor *y = nullptr;
    aclTensor *sortedIndex = nullptr;

    ret = CreateAclTensor(xHost, shape, &xDev, ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = CreateAclTensor(indexHost, shape, &indexDev, ACL_INT32, &index);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = CreateAclTensor(yHost, shape, &yDev, ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = CreateAclTensor(sortedIndexHost, shape, &sortedIndexDev, ACL_INT32, &sortedIndex);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    // ---- 2. 第一段接口：GetWorkspaceSize（入参校验 + workspace 大小 + executor）----
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSortWithIndexGetWorkspaceSize(x, index, axis, descending, stable, y, sortedIndex, &workspaceSize,
                                             &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSortWithIndexGetWorkspaceSize failed. ERROR: %d\n", ret); return -1);

    // ---- 3. 申请 workspace ----
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return -1);
    }

    // ---- 4. 第二段接口：执行计算 ----
    ret = aclnnSortWithIndex(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSortWithIndex failed. ERROR: %d\n", ret); return -1);

    // ---- 5. 同步并取回结果 ----
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return -1);

    ret = aclrtMemcpy(yHost.data(), yHost.size() * sizeof(float), yDev, n * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y failed. ERROR: %d\n", ret); return -1);
    ret = aclrtMemcpy(sortedIndexHost.data(), sortedIndexHost.size() * sizeof(int32_t), sortedIndexDev,
                      n * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy sorted_index failed. ERROR: %d\n", ret); return -1);

    // ---- 6. 打印结果（期望 y=[1,1.5,2,3,4]  sorted_index=[1,3,4,0,2]）----
    LOG_PRINT("y            = [");
    for (int64_t i = 0; i < n; i++) {
        LOG_PRINT("%.1f%s", yHost[i], (i + 1 < n) ? ", " : "");
    }
    LOG_PRINT("]\nsorted_index = [");
    for (int64_t i = 0; i < n; i++) {
        LOG_PRINT("%d%s", sortedIndexHost[i], (i + 1 < n) ? ", " : "");
    }
    LOG_PRINT("]\n");

    // ---- 7. 释放资源 ----
    aclDestroyTensor(x);
    aclDestroyTensor(index);
    aclDestroyTensor(y);
    aclDestroyTensor(sortedIndex);
    aclrtFree(xDev);
    aclrtFree(indexDev);
    aclrtFree(yDev);
    aclrtFree(sortedIndexDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
