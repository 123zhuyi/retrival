# aclnnAsStrided

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：根据输入张量 `x`、输出形状 `size`、步长 `stride` 和存储偏移 `storage_offset`，生成一个按照指定步长访问输入存储的输出张量，对应 PyTorch `torch.as_strided` 语义。

- 计算公式：

  $$
  y_i=x_{\text{storage\_offset}+\sum_{d=0}^{D-1}(i_d\cdot \text{stride}[d])}
  $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用 “aclnnAsStridedGetWorkspaceSize” 接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用 “aclnnAsStrided” 接口执行计算。

```cpp
aclnnStatus aclnnAsStridedGetWorkspaceSize(
    const aclTensor*   x,
    const aclIntArray* size,
    const aclIntArray* stride,
    const aclIntArray* storageOffset,
    aclTensor*         out,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAsStrided(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnAsStridedGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 245px">
  <col style="width: 340px">
  <col style="width: 330px">
  <col style="width: 110px">
  <col style="width: 140px">
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
      <td>输入张量。</td>
      <td>用于提供被访问的底层存储。</td>
      <td>INT64、UINT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BF16、INT16、UINT16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出张量的形状。</td>
      <td>长度必须与stride一致，且长度范围为1到8。</td>
      <td>INT64、INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>输出张量各维度映射到输入存储时使用的步长。</td>
      <td>长度必须与size一致；元素必须为非负整数。</td>
      <td>INT64、INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>storageOffset</td>
      <td>输入</td>
      <td>输出首元素相对于输入存储起始位置的偏移量。</td>
      <td>当前调用方式需要传入长度为1的aclIntArray；取值必须为非负整数。</td>
      <td>INT64、INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>shape需要与size指定的形状一致，数据类型需要与x一致。</td>
      <td>INT64、UINT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BF16、INT16、UINT16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>-</td>
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
      <td>传入的x、size、stride、storageOffset、out、workspaceSize或executor为空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>x和out的数据类型不在支持范围内，或二者数据类型不一致。</td>
    </tr>
    <tr>
      <td>size、stride或storageOffset的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>size与stride长度不一致，或长度不在1到8范围内。</td>
    </tr>
    <tr>
      <td>size、stride或storageOffset中存在负数。</td>
    </tr>
    <tr>
      <td>storageOffset长度不为1。</td>
    </tr>
    <tr>
      <td>根据size、stride和storageOffset访问的最大输入偏移超出x的存储范围。</td>
    </tr>
  </tbody>
  </table>

## aclnnAsStrided

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAsStridedGetWorkspaceSize获取。</td>
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
- `size` 和 `stride` 的长度必须一致，且长度范围为 1 到 8。
- `size`、`stride`、`storageOffset` 中的元素必须为非负整数。
- 当输出元素个数不为0时，`storageOffset + sum((size[d] - 1) * stride[d])` 不能超出输入张量存储范围。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <vector>
#include "acl/acl.h"
#include "aclnn_as_strided.h"

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    std::vector<int64_t> xShape = {10};
    std::vector<int64_t> yShape = {4};
    std::vector<int64_t> xStride = {1};
    std::vector<int64_t> yStride = {1};
    std::vector<int64_t> sizeData = {4};
    std::vector<int64_t> strideData = {2};
    std::vector<int64_t> storageOffsetData = {1};

    void* xDevice = nullptr;
    void* yDevice = nullptr;
    aclrtMalloc(&xDevice, xShape[0] * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&yDevice, yShape[0] * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);

    aclTensor* x = aclCreateTensor(xShape.data(), xShape.size(), ACL_INT32, xStride.data(), 0,
                                   ACL_FORMAT_ND, xShape.data(), xShape.size(), xDevice);
    aclTensor* y = aclCreateTensor(yShape.data(), yShape.size(), ACL_INT32, yStride.data(), 0,
                                   ACL_FORMAT_ND, yShape.data(), yShape.size(), yDevice);
    aclIntArray* size = aclCreateIntArray(sizeData.data(), sizeData.size());
    aclIntArray* stride = aclCreateIntArray(strideData.data(), strideData.size());
    aclIntArray* storageOffset = aclCreateIntArray(storageOffsetData.data(), storageOffsetData.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAsStridedGetWorkspaceSize(x, size, stride, storageOffset, y, &workspaceSize, &executor);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnAsStrided(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyIntArray(size);
    aclDestroyIntArray(stride);
    aclDestroyIntArray(storageOffset);
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

更多完整示例请参考 [test_aclnn_as_strided.cpp](../examples/test_aclnn_as_strided.cpp)。
