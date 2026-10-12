# aclnnStridedSliceGrad

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/strided_slice_grad)

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

- 接口功能：StridedSlice的反向算子，将切片梯度张量dy映射回原始张量形状shape的对应位置，未被切片覆盖的位置填零。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnStridedSliceGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnStridedSliceGrad”接口执行计算。

```Cpp
aclnnStatus aclnnStridedSliceGradGetWorkspaceSize(
    const aclIntArray  *shape,
    const aclIntArray  *begin,
    const aclIntArray  *end,
    const aclIntArray  *strides,
    const aclTensor    *dy,
    int64_t             beginMask,
    int64_t             endMask,
    int64_t             ellipsisMask,
    int64_t             newAxisMask,
    int64_t             shrinkAxisMask,
    const aclTensor    *out,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnStridedSliceGrad(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnStridedSliceGradGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
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
      <td>shape（aclIntArray*）</td>
      <td>输入</td>
      <td>原始输入张量的形状，即输出out的shape。</td>
      <td><ul><li>元素个数等于out的维度数。</li><li>各元素值大于0。</li><li>与begin/end/strides共同决定dy的shape，推导方法见dy参数说明。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>begin（aclIntArray*）</td>
      <td>输入</td>
      <td>切片起始位置索引，对应正向算子StridedSlice的begin参数。</td>
      <td><ul><li>长度必须与shape相同。</li><li>支持负索引（表示从末尾倒数）。</li><li>与shape/end/strides共同决定dy的shape，推导方法见dy参数说明。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end（aclIntArray*）</td>
      <td>输入</td>
      <td>切片结束位置索引（不含），对应正向算子StridedSlice的end参数。</td>
      <td><ul><li>长度必须与shape相同。</li><li>支持负索引。</li><li>与shape/begin/strides共同决定dy的shape，推导方法见dy参数说明。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides（aclIntArray*）</td>
      <td>输入</td>
      <td>切片步长，对应正向算子StridedSlice的strides参数。</td>
      <td><ul><li>长度必须与shape相同。</li><li>元素值不能为0。</li><li>支持负步长（反向切片）。</li><li>正步长时begin须小于等于end，负步长时begin须大于等于end，否则该维视为空切片。</li><li>被shrinkAxisMask置位的维度，其strides必须为1（该维只取begin指向的单个元素，步长无意义）。</li><li>与shape/begin/end共同决定dy的shape，推导方法见dy参数说明。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dy（aclTensor*）</td>
      <td>输入</td>
      <td>切片操作的输出梯度张量。</td>
      <td><ul>
        <li>dy的shape必须等于用shape、begin、end、strides及beginMask、endMask、ellipsisMask、newAxisMask、shrinkAxisMask对shape做切片后得到的张量形状。</li>
        <li>判断dy的shape是否正确的办法：把begin、end、strides按“索引项”与shape的各维逐项对齐（第i个元素对应第i项），再逐项统计切片取多少个元素，dy对应位置就有多大。注意newAxisMask/shrinkAxisMask置位的项会改变begin/end/strides与数据维的对齐关系（见下）。具体来说：
          <ul>
            <li>某项若被<code>shrinkAxisMask</code>置位，对应数据维只取begin指向的1个元素并从dy中降维（不计入dy的维数），该项消耗一个begin/end/strides元素；</li>
            <li>某项若被<code>newAxisMask</code>置位，dy在该位置多出一个长度为1的维度；该项会占用一个begin/end/strides元素但不作用于shape的任何数据维（即其后的begin/end/strides依次对齐到后续数据维）；</li>
            <li>若<code>ellipsisMask</code>置位，中间省略的若干维度全部保留（等效只切未省略的维度）；</li>
            <li>其余维度按“从begin开始，每个strides步取一个元素，取到end之前为止”统计个数，若取不到任何元素则该维大小为0；</li>
            <li><code>beginMask</code>/<code>endMask</code>置位的维度等同于不限制起点/终点，即取满整个维度。</li>
          </ul>
        </li>
      </ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beginMask（int64_t）</td>
      <td>输入</td>
      <td>位掩码，第i位为1时该维度的begin被忽略，从该维首元素开始。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>endMask（int64_t）</td>
      <td>输入</td>
      <td>位掩码，第i位为1时该维度的end被忽略，切片延伸至该维末尾。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ellipsisMask（int64_t）</td>
      <td>输入</td>
      <td>位掩码，标记省略号维度，被标记的维度全量保留（等价于":"）。</td>
      <td><ul><li>至多有1位为1。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>newAxisMask（int64_t）</td>
      <td>输入</td>
      <td>位掩码，第i位为1时在对应位置插入新维度（长度为1）。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shrinkAxisMask（int64_t）</td>
      <td>输入</td>
      <td>位掩码，第i位为1时对应维度在输出中被压缩（该维取单个元素）。被置位维度的strides必须为1。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>反向传播结果张量，shape与原始正向算子输入相同（即shape参数指定的形状）。</td>
      <td><ul><li>数据类型与dy一致。</li><li>未被切片覆盖的位置填0。</li></ul></td>
      <td>与dy一致</td>
      <td>ND</td>
      <td>-</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>dy、out存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>dy或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>shape、begin、end、strides的长度不一致。</td>
    </tr>
    <tr>
      <td>strides中存在值为0的元素。</td>
    </tr>
    <tr>
      <td>ellipsisMask中超过1位被置为1。</td>
    </tr>
  </tbody></table>

## aclnnStridedSliceGrad

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnStridedSliceGradGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnStridedSliceGrad默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_strided_slice_grad.h"

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
    // 固定写法，AscendCL 初始化
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 将 host 数据拷贝到 device
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 创建 aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream 初始化，参考 AscendCL 对外接口列表
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    //
    // 场景说明：
    //   正向 StridedSlice 从 shape=[3,4,5] 的张量中切出 [0:2, 1:3, 2:4]（步长全1），
    //   正向输出 shape = [2, 2, 2]。
    //   本反向算子将 dy（shape=[2,2,2]）中的梯度散射回 output（shape=[3,4,5]），
    //   未被切片覆盖的位置填 0。
    //
    // 参数：
    //   shape   = [3, 4, 5]   — 原始正向输入的 shape，即 output 的 shape
    //   begin   = [0, 1, 2]
    //   end     = [2, 3, 4]
    //   strides = [1, 1, 1]
    //   mask 全 0

    std::vector<int64_t> shapeData = {3, 4, 5};
    std::vector<int64_t> beginData = {0, 1, 2};
    std::vector<int64_t> endData = {2, 3, 4};
    std::vector<int64_t> stridesData = {1, 1, 1};

    std::vector<int64_t> dyShape = {2, 2, 2};
    std::vector<int64_t> outputShape = {3, 4, 5};

    // dy 梯度数据（float，全 1.0）
    std::vector<float> dyHostData(GetShapeSize(dyShape), 1.0f);
    // output 初始化为 0
    std::vector<float> outputHostData(GetShapeSize(outputShape), 0.0f);

    void* dyDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* output = nullptr;

    // 创建 dy aclTensor
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建值依赖输入（aclIntArray）
    aclIntArray* shape = aclCreateIntArray(shapeData.data(), shapeData.size());
    aclIntArray* begin = aclCreateIntArray(beginData.data(), beginData.size());
    aclIntArray* end = aclCreateIntArray(endData.data(), endData.size());
    aclIntArray* strides = aclCreateIntArray(stridesData.data(), stridesData.size());
    CHECK_RET(shape != nullptr && begin != nullptr && end != nullptr && strides != nullptr,
              LOG_PRINT("aclCreateIntArray failed.\n");
              return -1);

    // mask 参数（全 0 表示不启用任何特殊切片语义）
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;

    // 3. 调用 CANN 算子库 API（两段式接口）
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用 aclnnStridedSliceGrad 第一段接口
    ret = aclnnStridedSliceGradGetWorkspaceSize(shape, begin, end, strides, dy, beginMask, endMask, ellipsisMask,
                                                newAxisMask, shrinkAxisMask, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnStridedSliceGrad 第二段接口
    ret = aclnnStridedSliceGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGrad failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 将结果从 device 侧拷贝回 host 侧并打印
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outputDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放 aclIntArray 和 aclTensor
    aclDestroyIntArray(shape);
    aclDestroyIntArray(begin);
    aclDestroyIntArray(end);
    aclDestroyIntArray(strides);
    aclDestroyTensor(dy);
    aclDestroyTensor(output);

    // 7. 释放 Device 资源
    aclrtFree(dyDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
