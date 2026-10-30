# aclnnAddN

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
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
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：对输入TensorList中的tensors进行主元素相加求和操作。

- 计算公式：

  $$
  out = tensors_{1} + tensors_{2} + \dots + tensors_{n}
  $$

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnAddNGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnAddN"接口执行计算。

```cpp
aclnnStatus aclnnAddNGetWorkspaceSize(
  const aclTensorList *tensors,
  aclTensor           *out,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnAddN(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnAddNGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
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
      <td>tensors（const aclTensorList*）</td>
      <td>输入</td>
      <td>输入TensorList，对输入tensors进行主元素相加求和操作。</td>
      <td><li>支持空TensorList，此时输出为空Tensor。</li><li>tensors中的Tensor需要满足shape一致。</li></td>
      <td>INT32，INT64，FLOAT16，BFLOAT16，FLOAT32</td>
      <td>ND</td>
      <td>1~8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出Tensor，存储求和结果。</td>
      <td><li>不支持空Tensor。</li><li>数据类型需与tensors中的Tensor保持一致。</li><li>shape需要与tensors中的Tensor的shape一致。</li></td>
      <td>数据类型与tensors保持一致。</td>
      <td>ND</td>
      <td>1~8</td>
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
    <td>tensors是空指针，或tensors中的某个Tensor是空指针，或out是空指针。</td>
  </tr>
  <tr>
    <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="5">161002</td>
    <td>tensors或out的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>tensors或out的shape维度超过8。</td>
  </tr>
  <tr>
    <td>tensors中的Tensor数据类型不一致。</td>
  </tr>
  <tr>
    <td>tensors中的Tensor shape不一致。</td>
  </tr>
  <tr>
    <td>当前NPU架构不支持此算子，仅支持ASCEND910B(A2)和ASCEND910_93(A3)系列。</td>
  </tr>
  </tbody></table>

## aclnnAddN

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
          <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
      </thead>
      <tbody>
          <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
          <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnAddNGetWorkspaceSize获取。</td></tr>
          <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
          <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
      </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnAddN默认确定性实现。

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 数据类型：支持INT32，INT64，FLOAT16，BFLOAT16，FLOAT32。
  - 数据格式：仅支持ND格式。
  - 参数Shape：输入Tensor维度范围为1~8，需要满足shape一致。
<!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_n.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)           \
    do {                                  \
        printf(message, ##__VA_ARGS__);   \
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
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host数据拷贝到device侧
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (固定写法) device/stream初始化，参考acl API手册
    int32_t deviceId = 0; // 根据实际的device填写deviceId
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape1 = {4, 2};
    std::vector<int64_t> inputShape2 = {4, 2};
    std::vector<int64_t> outputShape = {4, 2};
    void* inputDeviceAddr1 = nullptr;
    void* inputDeviceAddr2 = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* output = nullptr;
    std::vector<float> inputHostData1 = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> inputHostData2 = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<float> outputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    aclDataType dtype = aclDataType::ACL_FLOAT;

    // 创建tensors的aclTensor
    ret = CreateAclTensor(inputHostData1, inputShape1, &inputDeviceAddr1, dtype, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData2, inputShape2, &inputDeviceAddr2, dtype, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out的aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, dtype, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建aclTensorList
    std::vector<aclTensor*> tmp{ input1, input2 };
    aclTensorList* inputList = aclCreateTensorList(tmp.data(), tmp.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用aclnnAddNGetWorkspaceSize第一段接口
    ret = aclnnAddNGetWorkspaceSize(inputList, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddNGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device侧内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnAddN第二段接口
    ret = aclnnAddN(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddN failed. ERROR: %d\n", ret); return ret);

    // 4. (固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧数据拷贝到host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclTensorList，需要根据具体API的接口定义修改
    aclDestroyTensor(output);
    aclDestroyTensorList(inputList);

    // 7. 释放device资源
    aclrtFree(inputDeviceAddr1);
    aclrtFree(inputDeviceAddr2);
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
