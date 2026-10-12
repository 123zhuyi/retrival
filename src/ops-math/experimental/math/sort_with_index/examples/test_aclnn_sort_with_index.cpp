/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_sort_with_index.cpp
 * \brief SortWithIndex (experimental, ascend910b) aclnn 两段式调用示例。
 *
 * 演示沿最后一维「升序」排序 + index 跟随重排的基础用例（float32 value + int32 index）：
 *   x     = [3.0, 1.0, 4.0, 1.5, 2.0]
 *   index = [  0,   1,   2,   3,   4]
 * 升序排序后期望（910B 语义，本例无 NaN/Inf）：
 *   y            = [1.0, 1.5, 2.0, 3.0, 4.0]
 *   sorted_index = [  1,   3,   4,   0,   2]
 *
 * 两段式接口：
 *   1) aclnnSortWithIndexGetWorkspaceSize  —— 入参校验 + workspace 大小 + executor
 *   2) aclnnSortWithIndex                  —— 执行计算
 *
 * 编译与运行：见同目录 run.sh（使用独立 vendor + ASCEND_CUSTOM_OPP_PATH 规避系统 built-in SortWithIndex）。
 */

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

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                           aclDataType dataType, aclTensor** tensor)
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

static int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

int main()
{
    // ---- 0. 初始化 ACL ----
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    // ---- 1. 构造输入（float32 value + int32 index），沿最后一维升序 ----
    std::vector<int64_t> shape = {5};
    std::vector<float> xHost = {3.0f, 1.0f, 4.0f, 1.5f, 2.0f};
    std::vector<int32_t> indexHost = {0, 1, 2, 3, 4};
    int64_t axis = -1;       // 仅支持最后一维
    bool descending = false;  // 升序
    bool stable = false;

    int64_t n = GetShapeSize(shape);
    std::vector<float> yHost(static_cast<size_t>(n), 0.0f);
    std::vector<int32_t> sortedIndexHost(static_cast<size_t>(n), 0);

    // 期望输出（CPU 参考，本例无 NaN/Inf）：升序 y=[1,1.5,2,3,4] sorted_index=[1,3,4,0,2]
    std::vector<float> yExpect = {1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
    std::vector<int32_t> sortedIndexExpect = {1, 3, 4, 0, 2};

    void* xDev = nullptr;
    void* indexDev = nullptr;
    void* yDev = nullptr;
    void* sortedIndexDev = nullptr;
    aclTensor* x = nullptr;
    aclTensor* index = nullptr;
    aclTensor* y = nullptr;
    aclTensor* sortedIndex = nullptr;

    ret = CreateAclTensor(xHost, shape, &xDev, ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = CreateAclTensor(indexHost, shape, &indexDev, ACL_INT32, &index);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = CreateAclTensor(yHost, shape, &yDev, ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = CreateAclTensor(sortedIndexHost, shape, &sortedIndexDev, ACL_INT32, &sortedIndex);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    // ---- 2. 第一段接口：GetWorkspaceSize ----
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSortWithIndexGetWorkspaceSize(x, index, axis, descending, stable, y, sortedIndex, &workspaceSize,
                                             &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSortWithIndexGetWorkspaceSize failed. ERROR: %d\n", ret); return -1);

    // ---- 3. 申请 workspace ----
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return -1);
    }

    // ---- 4. 第二段接口：执行 ----
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

    // ---- 6. 打印 + 校验 ----
    LOG_PRINT("==== SortWithIndex aclnn 升序基础用例 (float32 + int32) ====\n");
    LOG_PRINT("x            = [3.0, 1.0, 4.0, 1.5, 2.0]\n");
    LOG_PRINT("index        = [0, 1, 2, 3, 4]\n");
    LOG_PRINT("y            = [");
    for (int64_t i = 0; i < n; i++) {
        LOG_PRINT("%.1f%s", yHost[i], (i + 1 < n) ? ", " : "");
    }
    LOG_PRINT("]\n");
    LOG_PRINT("sorted_index = [");
    for (int64_t i = 0; i < n; i++) {
        LOG_PRINT("%d%s", sortedIndexHost[i], (i + 1 < n) ? ", " : "");
    }
    LOG_PRINT("]\n");

    bool pass = true;
    for (int64_t i = 0; i < n; i++) {
        if (yHost[i] != yExpect[i] || sortedIndexHost[i] != sortedIndexExpect[i]) {
            pass = false;
            LOG_PRINT("[MISMATCH] idx=%ld y=%.4f(exp %.4f) sorted_index=%d(exp %d)\n", static_cast<long>(i),
                      yHost[i], yExpect[i], sortedIndexHost[i], sortedIndexExpect[i]);
        }
    }
    LOG_PRINT("结果校验: %s\n", pass ? "PASS" : "FAIL");

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

    return pass ? 0 : -1;
}
