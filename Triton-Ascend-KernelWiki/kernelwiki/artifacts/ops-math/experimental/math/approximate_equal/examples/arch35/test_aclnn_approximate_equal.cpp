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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn_approximate_equal.h"

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

#define CHECK_ACL(expr)                                                                   \
    do {                                                                                  \
        aclError _err = (expr);                                                           \
        if (_err != ACL_SUCCESS) {                                                        \
            LOG_PRINT("[FAIL] %s returned %d at %s:%d", #expr, _err, __FILE__, __LINE__); \
            return _err;                                                                  \
        }                                                                                 \
    } while (0)

#define CHECK_RET(expr, ret_val)                                        \
    do {                                                                \
        if (!(expr)) {                                                  \
            LOG_PRINT("[FAIL] %s at %s:%d", #expr, __FILE__, __LINE__); \
            return (ret_val);                                           \
        }                                                               \
    } while (0)

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    if (shape.empty())
        return {};
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
static int CreateAclTensor(
    const std::vector<T>& host_data, const std::vector<int64_t>& shape, aclDataType dtype, void** device_addr,
    aclTensor** tensor)
{
    int64_t elem = 1;
    for (auto d : shape)
        elem *= d;
    size_t bytes = static_cast<size_t>(elem) * sizeof(T);
    if (bytes == 0)
        bytes = sizeof(T);

    CHECK_ACL(aclrtMalloc(device_addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!host_data.empty()) {
        CHECK_ACL(aclrtMemcpy(
            *device_addr, bytes, host_data.data(), host_data.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dtype, strides.empty() ? nullptr : strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *device_addr);
    CHECK_RET(*tensor != nullptr, ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

int main()
{
    LOG_PRINT("========================================");
    LOG_PRINT("aclnnApproximateEqual 调用示例");
    LOG_PRINT("========================================");

    const int32_t device_id = 0;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ------------------------------------------------------------------------
    // 准备输入/输出 Host 侧数据
    // ------------------------------------------------------------------------
    const std::vector<int64_t> shape = {8};
    const float tolerance = 1e-5f;

    std::vector<float> x1_host = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> x2_host = {1.0f, 2.1f, 3.0f, 4.5f, 5.0f, 6.0f, 7.0f, 9.0f};
    std::vector<uint8_t> y_host(shape[0], 0);

    // 预期结果（|x1-x2| < 1e-5）
    const std::vector<uint8_t> expected = {1, 0, 1, 0, 1, 1, 1, 0};

    // ------------------------------------------------------------------------
    // 创建 aclTensor
    // ------------------------------------------------------------------------
    void* x1_dev = nullptr;
    void* x2_dev = nullptr;
    void* y_dev = nullptr;
    aclTensor* x1_t = nullptr;
    aclTensor* x2_t = nullptr;
    aclTensor* y_t = nullptr;

    CHECK_ACL(CreateAclTensor(x1_host, shape, ACL_FLOAT, &x1_dev, &x1_t));
    CHECK_ACL(CreateAclTensor(x2_host, shape, ACL_FLOAT, &x2_dev, &x2_t));
    CHECK_ACL(CreateAclTensor(y_host, shape, ACL_BOOL, &y_dev, &y_t));

    // ------------------------------------------------------------------------
    // 两段式接口：GetWorkspaceSize -> launch
    // ------------------------------------------------------------------------
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnApproximateEqualGetWorkspaceSize(x1_t, x2_t, tolerance, y_t, &ws_size, &executor);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    LOG_PRINT("GetWorkspaceSize OK, workspaceSize = %llu", static_cast<unsigned long long>(ws_size));

    void* workspace = nullptr;
    if (ws_size > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ret = aclnnApproximateEqual(workspace, ws_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // ------------------------------------------------------------------------
    // 读回 & 比对 & 打印
    // ------------------------------------------------------------------------
    std::vector<uint8_t> y_out(shape[0], 0);
    CHECK_ACL(aclrtMemcpy(y_out.data(), y_out.size(), y_dev, y_out.size(), ACL_MEMCPY_DEVICE_TO_HOST));

    LOG_PRINT("NPU 输出 y = [");
    for (size_t i = 0; i < y_out.size(); ++i) {
        LOG_PRINT("  [%zu] x1=%.5f x2=%.5f => %s", i, x1_host[i], x2_host[i], (y_out[i] != 0) ? "true" : "false");
    }
    LOG_PRINT("]");

    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        uint8_t a = (y_out[i] != 0) ? 1 : 0;
        if (a != expected[i]) {
            LOG_PRINT("不匹配 [%zu] expected=%u got=%u", i, (unsigned)expected[i], (unsigned)a);
            ok = false;
        }
    }

    // ------------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------------
    if (workspace)
        aclrtFree(workspace);
    aclDestroyTensor(x1_t);
    aclDestroyTensor(x2_t);
    aclDestroyTensor(y_t);
    aclrtFree(x1_dev);
    aclrtFree(x2_dev);
    aclrtFree(y_dev);

    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();

    LOG_PRINT("========================================");
    if (ok) {
        LOG_PRINT("PASS");
        return 0;
    }
    LOG_PRINT("FAIL");
    return 1;
}
