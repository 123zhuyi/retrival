# aclnnNeTensor&aclnnInplaceNeTensor

## 产品支持情况

| 产品                                           | 是否支持 |
|:---------------------------------------------|:----:|
| <term>Ascend 950PR/Ascend 950DT</term>       |  ×   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |  ×   |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |  √   |
| <term>Atlas 200I/500 A2 推理产品</term>          |  ×   |
| <term>Atlas 推理系列产品</term>                    |  ×   |
| <term>Atlas 训练系列产品</term>                    |  ×   |

## 功能说明

* 算子功能：计算self（selfRef）中的元素的值与other的值是否不相等。
* 计算公式：

$$
out_i=(self_i != other_i)?[1]:[0]
$$

$$
selfRef_i=(selfRef_i\ != other_i)\ ?\  [1]:[0]
$$

## 函数原型

* aclnnNeTensor和aclnnInplaceNeTensor实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  * aclnnNeTensor：需新建一个输出张量对象存储计算结果。
  * aclnnInplaceNeTensor：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
* 每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnNeTensorGetWorkspaceSize”或者“aclnnInplaceNeTensorGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnNeTensor”或者“aclnnInplaceNeTensor”接口执行计算。
  * `aclnnStatus aclnnNeTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnNeTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceNeTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceNeTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnNeTensorGetWorkspaceSize

* **参数说明**：
  - self(aclTensor*, 计算输入)：公式中的`self`，Device侧的aclTensor，数据类型需要与other满足数据类型推导规则（参见[互推导关系](../../../../docs/zh/context/deduction_relationship.md)），shape需要与other满足[broadcast关系](../../../../docs/zh/context/broadcast_relationship.md)，shape维度不高于8维。支持[非连续的Tensor](../../../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../../../docs/zh/context/data_format.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、
    UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
  - other(aclTensor*, 计算输入)：公式中的`other`，Device侧的aclTensor，数据类型需要与self满足数据类型推导规则（参见[互推导关系](../../../../docs/zh/context/deduction_relationship.md)），shape需要与self的shape满足[broadcast关系](../../../../docs/zh/context/broadcast_relationship.md)，shape维度不高于8维。支持[非连续的Tensor](../../../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../../../docs/zh/context/data_format.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
  - out(aclTensor \*, 计算输出)：公式中的`out`，Device侧的aclTensor，数据类型需要是BOOL可转换的数据类型[互转换关系](../../../../docs/zh/context/conversion_relationship.md), shape与self、other广播之后的shape（参见[broadcast关系](../../../../docs/zh/context/broadcast_relationship.md)）一致，shape维度不高于8维，支持[非连续的Tensor](../../../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../../../docs/zh/context/data_format.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64、UINT32、UINT16。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
  + workspaceSize（uint64\_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  + executor(aclOpExecutor\*\*, 出参)：返回op执行器，包含了算子计算流程。
* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  ````text
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、other、out是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、other或out的数据类型不在支持的范围之内。
                                        2. self、other或out的维度大于8。
                                        3. self和other的数据类型无法进行推导。
                                        4. self和other的shape无法进行broadcast。
                                        5. out的shape与broadcast后的shape不一致。
  ````

## aclnnNeTensor

* **参数说明**：
  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnNeTensorGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。
* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## aclnnInplaceNeTensorGetWorkspaceSize

* **参数说明**：
  * selfRef（aclTensor*, 计算输入/输出）：公式中的`selfRef`。数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../../docs/zh/context/deduction_relationship.md)），shape需要与other满足[broadcast关系](../../../../docs/zh/context/broadcast_relationship.md)。支持[非连续的Tensor](../../../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../../../docs/zh/context/data_format.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
  * other（aclTensor*, 计算输入）：公式中的`other`。数据类型与selfRef的数据类型需满足数据类型推导规则（参见[互推导关系](../../../../docs/zh/context/deduction_relationship.md)），shape需要与selfRef满足[broadcast关系](../../../../docs/zh/context/broadcast_relationship.md)。支持[非连续的Tensor](../../../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../../../docs/zh/context/data_format.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。

  + workspaceSize(uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。
  + executor(aclOpExecutor\*\*, 出参)：返回op执行器，包含了算子计算流程。
* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

```text
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的selfRef、other是空指针时。
返回161002（ACLNN_ERR_PARAM_INVALID）：1. selfRef和other的数据类型不在支持的范围之内。
                                      2. selfRef和other的数据类型无法进行推导。
                                      3. selfRef和other的shape无法做broadcast。
                                      4. selfRef和other做broadcast后的shape不等于selfRef的shape。
                                      5. selfRef、other的维度大于8。
```

## aclnnInplaceNeTensor

* **参数说明**：
  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnInplaceNeTensorGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。
* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_ne_tensor.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void *selfDeviceAddr = nullptr;
    void *otherDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other aclTensor
    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnNeTensor接口调用示例
    // 3. 调用CANN算子库API
    LOG_PRINT("test aclnnNeTensor\n");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnNeTensor第一段接口
    ret = aclnnNeTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnNeTensor第二段接口
    ret = aclnnNeTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeTensor failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    //aclnnInplaceNeTensor接口调用示例
    // 3. 调用CANN算子库API
    LOG_PRINT("\ntest aclnnInplaceNeTensor\n");
    // 调用aclnnInplaceNeTensor第一段接口
    ret = aclnnInplaceNeTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnInplaceNeTensor第二段接口
    ret = aclnnInplaceNeTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeTensor failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(other);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    if (workspaceSize > 0) {
       aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
