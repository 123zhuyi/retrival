/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_isin_part_v1.h"

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

// 获取Shape对应的元素总数
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
        shapeSize *= i;
    return shapeSize;
}

// 仅创建设备内存和Tensor句柄，不拷贝数据
int CreateEmptyTensor(const std::vector<int64_t>& shape, aclDataType dataType, void** deviceAddr, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 创建aclTensor
template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 结果输出
void PrintOutResult(std::vector<int64_t>& shape, int32_t elementsNum, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("[Output] Result Vector: [");
    for (int64_t i = 0; i < elementsNum; i++) {
        // 使用 %u 打印无符号整数
        LOG_PRINT("%u%s", resultData[i], (i == elementsNum - 1) ? "" : ", ");
    }
    LOG_PRINT("]\n");
}

int main()
{
    // 1. 环境初始化
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateContext(&context, deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

    // 2. 直接构造 IsinPartV1 的三个输入，仅校验本算子（不依赖 Cat/Sort/Cast 等其它算子）。
    //    语义等价于 isin(elements=[1,1,4,1,1], test_elements=[0,1,2,3])：
    //    - 将 elements 与 test_elements 拼接后按值排序，相等值相邻；
    //    - value 为排序后的值，index 为其在拼接序列中的原始位置（0..elementsNum-1 为待判定元素）；
    //    - elementsNum 为待判定元素个数（单元素 INT32 张量）；
    //    - 输出 z[i] 表示第 i 个待判定元素是否出现在 test_elements 中。
    std::vector<float> h_value = {0, 1, 1, 1, 1, 1, 2, 3, 4}; // 排序后的值
    std::vector<int32_t> h_index = {5, 0, 1, 3, 4, 6, 7, 8, 2}; // 对应的原始位置
    int32_t elementsNum = 5;
    std::vector<int32_t> h_elementsNum = {elementsNum};
    int64_t totalSize = static_cast<int64_t>(h_value.size());

    std::vector<int64_t> totalShape = {totalSize};
    std::vector<int64_t> outShape = {elementsNum};

    // 3. 资源准备
    void *valueAddr = nullptr, *indexAddr = nullptr, *elementsNumAddr = nullptr, *outAddr = nullptr;
    aclTensor *valueTensor = nullptr, *indexTensor = nullptr, *elementsNumTensor = nullptr, *outTensor = nullptr;
    CHECK_RET(CreateAclTensor(h_value, totalShape, &valueAddr, ACL_FLOAT, &valueTensor) == 0, return -1);
    CHECK_RET(CreateAclTensor(h_index, totalShape, &indexAddr, ACL_INT32, &indexTensor) == 0, return -1);
    CHECK_RET(CreateAclTensor(h_elementsNum, {1}, &elementsNumAddr, ACL_INT32, &elementsNumTensor) == 0, return -1);
    CHECK_RET(CreateEmptyTensor(outShape, ACL_BOOL, &outAddr, &outTensor) == 0, return -1);

    // 4. 调用自定义算子 aclnnIsinPartV1
    LOG_PRINT("[Process] aclnnIsinPartV1\n");
    uint64_t wsIsin = 0;
    aclOpExecutor* execIsin = nullptr;
    CHECK_RET(
        aclnnIsinPartV1GetWorkspaceSize(valueTensor, indexTensor, elementsNumTensor, outTensor, &wsIsin, &execIsin) ==
            ACL_SUCCESS,
        return -1);
    void* wsIsinAddr = nullptr;
    if (wsIsin > 0)
        aclrtMalloc(&wsIsinAddr, wsIsin, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclnnIsinPartV1(wsIsinAddr, wsIsin, execIsin, stream) == ACL_SUCCESS, return -1);

    // 5. 同步并获取结果（期望 [1, 1, 0, 1, 1]，即 1 在测试集中、4 不在）
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);
    LOG_PRINT("--- Final Result ---\n");
    PrintOutResult(outShape, elementsNum, &outAddr);

    // 6. 资源清理
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(indexTensor);
    aclDestroyTensor(elementsNumTensor);
    aclDestroyTensor(outTensor);

    aclrtFree(valueAddr);
    aclrtFree(indexAddr);
    aclrtFree(elementsNumAddr);
    aclrtFree(outAddr);
    if (wsIsinAddr)
        aclrtFree(wsIsinAddr);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();

    return 0;
}
