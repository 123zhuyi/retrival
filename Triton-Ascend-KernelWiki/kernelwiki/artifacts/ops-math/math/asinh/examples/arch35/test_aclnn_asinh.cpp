/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_aclnn_asinh.cpp
 * @brief Asinh 算子 ACLNN 两段式接口调用示例（Ascend950 DAV_3510 / arch35）
 *
 * 演示流程（最小可运行示例）：
 *   1. aclInit + aclrtSetDevice + aclrtCreateStream 初始化 ACL 资源
 *   2. 构造 host 端 FP32 输入并搬运到 device，构造 input / out aclTensor
 *   3. 两段式调用：
 *        aclnnAsinhGetWorkspaceSize(input, out, &workspaceSize, &executor)
 *        aclrtMalloc(workspaceAddr, workspaceSize, ...)
 *        aclnnAsinh(workspaceAddr, workspaceSize, executor, stream)
 *   4. aclrtSynchronizeStream + 拷回 host
 *   5. 与 std::asinh 标杆逐元素比对，包含 asinh(1.0) ≈ 0.881373587 验证点
 *   6. 释放资源（Tensor / Memory / Stream / Device / Acl）
 *
 * 编译运行：
 *   1. source <cann_install>/set_env.sh
 *   2. cd operators/asinh/examples && bash run.sh
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "acl/acl.h"
#include "aclnn_asinh.h"  // 来自 vendor 包 opp/vendors/asinh_custom/op_api/include/

// ---- 简易宏 ----
#define CHECK_RET(cond, return_expr) \
    do {                              \
        if (!(cond)) {                \
            return_expr;              \
        }                             \
    } while (0)

#define LOG_PRINT(message, ...)       \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape) {
        size *= d;
    }
    return size;
}

// ---- ACL 初始化（device + stream）----
static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);

    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    return 0;
}

// ---- 构造 aclTensor (FP32 / ND，连续) ----
template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData,
                           const std::vector<int64_t>& shape,
                           void** deviceAddr,
                           aclDataType dataType,
                           aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. ACL device / stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[INFO] ACL initialized. deviceId=%d\n", deviceId);

    // 2. 构造输入 / 输出
    //   shape = [16]，包含负值、0、正值，并覆盖关键比对点 1.0（期望 asinh(1.0) ≈ 0.881373587）
    std::vector<int64_t> selfShape = {16};
    std::vector<int64_t> outShape  = {16};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr  = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out  = nullptr;

    std::vector<float> selfHostData = {
        -4.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.1f, -0.0001f, -0.0f,
         0.0f,  0.0001f, 0.1f,  0.5f,  1.0f,  1.5f,  2.0f,   4.0f};
    std::vector<float> outHostData(selfHostData.size(), 0.0f);

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData,  outShape,  &outDeviceAddr,  aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    LOG_PRINT("[INFO] Input tensors built. numel=%ld dtype=FP32 shape=[16]\n",
              GetShapeSize(selfShape));

    // 3. 两段式接口 - 第一段：获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAsinhGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnAsinhGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[INFO] aclnnAsinhGetWorkspaceSize OK. workspaceSize=%lu\n", workspaceSize);

    // 4. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 两段式接口 - 第二段：执行算子
    ret = aclnnAsinh(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnAsinh failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待执行完成
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[INFO] aclnnAsinh kernel executed on NPU.\n");

    // 7. 取回结果
    auto outNumel = GetShapeSize(outShape);
    std::vector<float> resultData(outNumel, 0.0f);
    ret = aclrtMemcpy(resultData.data(), outNumel * sizeof(float),
                      outDeviceAddr,     outNumel * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);

    // 8. 与 std::asinh 标杆比对（FP32 双万：atol = rtol = 1e-4）
    //    关键比对点：asinh(1.0) ≈ 0.881373587
    const float atol = 1e-4f;
    const float rtol = 1e-4f;
    int passCount = 0;
    int failCount = 0;
    float maxAtol = 0.0f;
    float maxRtol = 0.0f;
    int idx_asinh1 = -1;

    LOG_PRINT("\n[RESULT] aclnnAsinh output vs std::asinh golden:\n");
    LOG_PRINT("  %-3s %-13s %-15s %-15s %-12s %-7s\n",
              "idx", "input", "npu_out", "golden", "abs_err", "pass");
    for (int64_t i = 0; i < outNumel; ++i) {
        float gold = std::asinh(selfHostData[i]);
        float absErr = std::fabs(resultData[i] - gold);
        float relErr = (std::fabs(gold) > 0.0f) ? absErr / std::fabs(gold) : 0.0f;
        maxAtol = std::max(maxAtol, absErr);
        maxRtol = std::max(maxRtol, relErr);
        bool ok = absErr <= atol + rtol * std::fabs(gold);
        if (ok) ++passCount; else ++failCount;
        if (std::fabs(selfHostData[i] - 1.0f) < 1e-9f) idx_asinh1 = static_cast<int>(i);
        LOG_PRINT("  %-3ld % .6f      % .10f   % .10f   %.2e    %s\n",
                  i, selfHostData[i], resultData[i], gold, absErr, ok ? "OK" : "FAIL");
    }
    LOG_PRINT("\n[SUMMARY] total=%ld pass=%d fail=%d  max_atol=%.3e  max_rtol=%.3e\n",
              outNumel, passCount, failCount, maxAtol, maxRtol);

    if (idx_asinh1 >= 0) {
        const float expected = 0.881373587f;
        float diff = std::fabs(resultData[idx_asinh1] - expected);
        LOG_PRINT("[ASSERT] asinh(1.0) = %.10f  (expected ≈ %.9f, diff=%.3e) %s\n",
                  resultData[idx_asinh1], expected, diff,
                  diff < 1e-4f ? "PASS" : "FAIL");
    }

    int exitCode = (failCount == 0) ? 0 : 1;
    LOG_PRINT("[FINAL] %s\n", exitCode == 0 ? "ALL PASS" : "FAILED");

    // 9. 资源释放
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return exitCode;
}
