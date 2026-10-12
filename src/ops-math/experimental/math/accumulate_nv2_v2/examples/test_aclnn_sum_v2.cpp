/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_sum_v2.cpp
 * @brief AccumulateNv2V2 算子 aclnn 调用示例
 *
 * 功能：使用 aclnnSumV2 两段式接口完成 N 个输入 tensor 的逐元素累加
 * 接口：aclnnSumV2GetWorkspaceSize + aclnnSumV2
 *
 * 编译方式：
 *   source /path/to/Ascend/cann/set_env.sh
 *   g++ -std=c++17 -o test_aclnn_sum_v2 test_aclnn_sum_v2.cpp \
 *       -I${ASCEND_HOME_PATH}/include \
 *       -I${ASCEND_HOME_PATH}/opp/vendors/accumulate_nv2_v2_custom/op_api/include \
 *       -L${ASCEND_HOME_PATH}/lib64 \
 *       -L${ASCEND_HOME_PATH}/opp/vendors/accumulate_nv2_v2_custom/op_api/lib \
 *       -lcust_opapi -lnnopbase -lopapi -lascendcl
 *
 * 运行方式：
 *   ./test_aclnn_sum_v2
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_sum_v2.h"

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
    int64_t size = 1;
    for (auto dim : shape)
        size *= dim;
    return size;
}

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

int main()
{
    // 1. 初始化 ACL 资源
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = aclnnInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInit failed. ERROR: %d\n", ret); return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return 1);

    // 2. 构造输入数据：3 个 float32 tensor，shape=[2,4]
    std::vector<int64_t> shape = {2, 4};
    int64_t numElements = GetShapeSize(shape);
    int N = 3;

    std::vector<std::vector<float>> hostInputs = {
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f},
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}};
    // 期望输出: {1.6, 3.7, 5.8, 7.9, 10.0, 12.1, 14.2, 16.3}

    std::vector<void*> inputDevs(N, nullptr);
    std::vector<aclTensor*> inputTensors(N, nullptr);

    for (int i = 0; i < N; i++) {
        ret = CreateAclTensor(hostInputs[i], shape, &inputDevs[i], ACL_FLOAT, &inputTensors[i]);
        CHECK_RET(ret == 0, LOG_PRINT("Create input tensor %d failed\n", i); return 1);
    }

    aclTensorList* tensorList = aclCreateTensorList(inputTensors.data(), N);
    CHECK_RET(tensorList != nullptr, LOG_PRINT("aclCreateTensorList failed\n"); return 1);

    // 3. 构造输出 tensor
    std::vector<float> outHostData(numElements, 0.0f);
    void* outDevAddr = nullptr;
    aclTensor* outTensor = nullptr;
    ret = CreateAclTensor(outHostData, shape, &outDevAddr, ACL_FLOAT, &outTensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create output tensor failed\n"); return 1);

    // 4. 调用 aclnnSumV2 两段式接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnSumV2GetWorkspaceSize(tensorList, outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSumV2GetWorkspaceSize failed. ERROR: %d\n", ret); return 1);
    LOG_PRINT("workspaceSize = %lu\n", workspaceSize);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Alloc workspace failed. ERROR: %d\n", ret); return 1);
    }

    ret = aclnnSumV2(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSumV2 failed. ERROR: %d\n", ret); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Sync stream failed. ERROR: %d\n", ret); return 1);

    // 5. 获取输出结果
    std::vector<float> result(numElements);
    ret = aclrtMemcpy(
        result.data(), numElements * sizeof(float), outDevAddr, numElements * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("D2H copy failed. ERROR: %d\n", ret); return 1);

    LOG_PRINT("\nAccumulateNv2V2 result (N=%d, shape=[2,4], float32):\n", N);
    for (int64_t i = 0; i < numElements; i++) {
        LOG_PRINT("  output[%ld] = %.4f\n", i, result[i]);
    }

    // 6. 释放资源
    if (workspace)
        aclrtFree(workspace);
    aclDestroyTensor(outTensor);
    aclrtFree(outDevAddr);
    aclDestroyTensorList(tensorList);
    for (int i = 0; i < N; i++) {
        aclrtFree(inputDevs[i]);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclnnFinalize();

    LOG_PRINT("\nTest completed successfully!\n");
    return 0;
}
