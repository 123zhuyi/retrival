# aclnnReduceMinV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：对输入张量沿指定维度计算最小值，对应 PyTorch `torch.min(input, dim)` 语义。计算结果写入输出张量，支持通过 `keepdims` 控制是否保留被 reduce 的维度。

- 计算公式：

  $$
  y = \min_{axis}(x)
  $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用 "aclnnReduceMinV2GetWorkspaceSize" 接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用 "aclnnReduceMinV2" 接口执行计算。

```cpp
aclnnStatus aclnnReduceMinV2GetWorkspaceSize(
    const aclTensor* x,
    int64_t          axes,
    int64_t          keepdims,
    aclTensor*       y,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnReduceMinV2(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnReduceMinV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 160px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 350px">
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 160px">
  <col style="width: 85px">
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
      <td>待计算最小值的输入张量。</td>
      <td>仅支持 2 维张量。</td>
      <td>float32、float16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>属性</td>
      <td>指定沿哪个轴进行 reduce 的最小值计算。</td>
      <td>仅支持单个轴，取值范围为 0 或 1。</td>
      <td>int64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdims</td>
      <td>属性</td>
      <td>是否保留被 reduce 的维度。</td>
      <td>0 表示不保留，输出维度数减少；1 表示保留，被 reduce 的维度置为 1。</td>
      <td>int64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>reduce 最小值的结果张量。</td>
      <td>dtype 需要与输入 `x` 一致。shape 由 `x`、`axes` 和 `keepdims` 决定。</td>
      <td>float32、float16</td>
      <td>ND</td>
      <td>1-2</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
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
      <td>传入的 x、y、workspaceSize 或 executor 为空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x 或 y 的数据类型不在 float32、float16 支持范围内，或二者数据类型不一致。</td>
    </tr>
    <tr>
      <td>x 的数据格式不支持（仅支持 ND）。</td>
    </tr>
    <tr>
      <td>x 的维度不为 2。</td>
    </tr>
    <tr>
      <td>axes 不在 [0, 1] 范围内。</td>
    </tr>
    <tr>
      <td>keepdims 不为 0 或 1。</td>
    </tr>
  </tbody>
  </table>

## aclnnReduceMinV2

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>在Device侧申请的workspace大小，由第一段接口 aclnnReduceMinV2GetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持 ND 格式。
- 输入 `x` 仅支持 2 维张量。
- `axes` 仅支持单个轴（0 或 1）。
- `keepdims` 仅支持 0 或 1。
- 数据类型仅支持 float32 和 float16。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <vector>
#include "acl/acl.h"
#include "aclnn_reduce_min_v2.h"

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    std::vector<int64_t> xShape = {4, 8};
    std::vector<int64_t> xStride = {8, 1};
    std::vector<int64_t> yShape = {1, 8};
    std::vector<int64_t> yStride = {8, 1};

    void* xDevice = nullptr;
    void* yDevice = nullptr;
    aclrtMalloc(&xDevice, 32 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&yDevice, 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclTensor* x = aclCreateTensor(xShape.data(), xShape.size(), ACL_FLOAT, xStride.data(), 0,
                                   ACL_FORMAT_ND, xShape.data(), xShape.size(), xDevice);
    aclTensor* y = aclCreateTensor(yShape.data(), yShape.size(), ACL_FLOAT, yStride.data(), 0,
                                   ACL_FORMAT_ND, yShape.data(), yShape.size(), yDevice);

    int64_t axes = 0;
    int64_t keepdims = 1;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnReduceMinV2GetWorkspaceSize(x, axes, keepdims, y, &workspaceSize, &executor);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnReduceMinV2(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

更多完整示例请参考 [test_aclnn_reduce_min_v2.cpp](../examples/test_aclnn_reduce_min_v2.cpp)。
