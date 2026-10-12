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
 * \file test_aclnn_arange.cpp
 * \brief aclnnArange 两段式调用示例。
 *
 * 覆盖多 dtype 代表用例：
 *   - 用例 1：FLOAT  升序（start=0, end=10, step=1）
 *   - 用例 2：FLOAT  负 step 降序（start=5, end=-5, step=-2）
 *   - 用例 3：INT8   窄整型升序（start=-3, end=12, step=3）
 *   - 用例 4/5：FLOAT 非有限值（+inf / nan）传播演示（评审意见：覆盖 inf/nan 场景）
 *
 * 元素个数 N 由调用方按 N = ceil((end - start) / step) 计算并据此构造 out 张量
 * （算子侧不计算 / 不校验 N，见 README 约束说明）。
 *
 * 非有限值说明：inf/nan 属 README“调用方前置约束”之外的值级输入（接口不做值级校验）。
 *   此类输入下 N 无法由 ceil((end-start)/step) 稳健推导，故由调用方显式给定一个小 N；
 *   算子按 IEEE 语义逐元素生成 out[i]=start+i*step（FLOAT 走纯 FP32 路径），
 *   inf/nan 按 IEEE 传播且不崩溃，由用例 4/5 演示。
 *
 * 编译运行：bash build.sh --run_example arange eager cust --vendor_name=custom --experimental
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "acl/acl.h"
#include "aclnn_arange.h"

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

// 创建一维连续输出 aclTensor（仅分配 device 内存，不需要拷入初值）
int CreateOutTensor(int64_t n, size_t elemSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = static_cast<size_t>(n) * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {n};
    std::vector<int64_t> strides = {1};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, return -1);
    return 0;
}

// 按 dtype 把 device 输出拷回 host 并打印
template <typename T>
void PrintTyped(const std::string& tag, int64_t n, void* deviceAddr, bool isFloat)
{
    std::vector<T> resultData(n);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), deviceAddr,
                           static_cast<size_t>(n) * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result D2H failed. ERROR: %d\n", ret); return);
    LOG_PRINT("[%s] N=%ld -> [", tag.c_str(), n);
    for (int64_t i = 0; i < n; i++) {
        if (isFloat) {
            LOG_PRINT("%g%s", static_cast<double>(resultData[i]), (i + 1 == n) ? "" : ", ");
        } else {
            LOG_PRINT("%ld%s", static_cast<long>(resultData[i]), (i + 1 == n) ? "" : ", ");
        }
    }
    LOG_PRINT("]\n");
}

// 计算 N = ceil((end - start) / step)（调用方职责）
int64_t ComputeN(double start, double end, double step)
{
    return static_cast<int64_t>(std::ceil((end - start) / step));
}

// 运行一组 aclnnArange 用例（FLOAT 路径，调用方显式给定 N）
// 用于 inf/nan 等非有限值演示：此类输入的 N 无法由 ceil((end-start)/step) 稳健推导，
// 故由调用方直接传入一个小 N，验证算子对非有限标量按 IEEE 语义逐元素传播、不崩溃。
int RunFloatCaseExplicitN(const std::string& tag, aclrtStream stream, float start, float end, float step, int64_t n)
{
    LOG_PRINT("---- %s (start=%g end=%g step=%g, N=%ld) ----\n", tag.c_str(), start, end, step, n);

    aclScalar* sStart = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* sEnd = aclCreateScalar(&end, aclDataType::ACL_FLOAT);
    aclScalar* sStep = aclCreateScalar(&step, aclDataType::ACL_FLOAT);
    CHECK_RET(sStart && sEnd && sStep, LOG_PRINT("create scalar failed\n"); return -1);

    void* outDeviceAddr = nullptr;
    aclTensor* out = nullptr;
    auto ret = CreateOutTensor(n, sizeof(float), &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == 0, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnArangeGetWorkspaceSize(sStart, sEnd, sStep, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArangeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnArange(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArange failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintTyped<float>(tag, n, outDeviceAddr, /*isFloat=*/true);

    aclDestroyScalar(sStart);
    aclDestroyScalar(sEnd);
    aclDestroyScalar(sStep);
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

// 运行一组 aclnnArange 用例（FLOAT 路径，N 由调用方按 ceil 公式自动计算）
int RunFloatCase(const std::string& tag, aclrtStream stream, float start, float end, float step)
{
    return RunFloatCaseExplicitN(tag, stream, start, end, step, ComputeN(start, end, step));
}

// 运行一组 aclnnArange 用例（INT8 路径）
int RunInt8Case(const std::string& tag, aclrtStream stream, int8_t start, int8_t end, int8_t step)
{
    int64_t n = ComputeN(start, end, step);
    LOG_PRINT("---- %s (start=%d end=%d step=%d, expect N=%ld) ----\n", tag.c_str(), start, end, step, n);

    aclScalar* sStart = aclCreateScalar(&start, aclDataType::ACL_INT8);
    aclScalar* sEnd = aclCreateScalar(&end, aclDataType::ACL_INT8);
    aclScalar* sStep = aclCreateScalar(&step, aclDataType::ACL_INT8);
    CHECK_RET(sStart && sEnd && sStep, LOG_PRINT("create scalar failed\n"); return -1);

    void* outDeviceAddr = nullptr;
    aclTensor* out = nullptr;
    auto ret = CreateOutTensor(n, sizeof(int8_t), &outDeviceAddr, aclDataType::ACL_INT8, &out);
    CHECK_RET(ret == 0, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnArangeGetWorkspaceSize(sStart, sEnd, sStep, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArangeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnArange(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArange failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintTyped<int8_t>(tag, n, outDeviceAddr, /*isFloat=*/false);

    aclDestroyScalar(sStart);
    aclDestroyScalar(sEnd);
    aclDestroyScalar(sStep);
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int main()
{
    // 1. device / stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int failed = 0;

    // 2. 多 dtype 代表用例
    // 用例 1：FLOAT 升序  -> [0, 1, 2, ..., 9]
    if (RunFloatCase("case1-FLOAT-asc", stream, 0.0f, 10.0f, 1.0f) != 0) {
        failed++;
    }
    // 用例 2：FLOAT 负 step 降序 -> [5, 3, 1, -1, -3]
    if (RunFloatCase("case2-FLOAT-neg-step", stream, 5.0f, -5.0f, -2.0f) != 0) {
        failed++;
    }
    // 用例 3：INT8 窄整型升序 -> [-3, 0, 3, 6, 9]
    if (RunInt8Case("case3-INT8-asc", stream, -3, 12, 3) != 0) {
        failed++;
    }

    // 用例 4/5：FLOAT 非有限值传播演示（覆盖 inf/nan 场景）
    // inf/nan 属 README“调用方前置约束”之外的值级输入，N 无法由 ceil 公式稳健推导，
    // 故显式指定 N=5；算子按 IEEE 语义逐元素生成 start+i*step（finite step），验证传播且不崩溃。
    const float kPosInf = std::numeric_limits<float>::infinity();
    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    // 用例 4：start=+inf, finite step -> 全 +inf（inf + i*step = inf）
    if (RunFloatCaseExplicitN("case4-FLOAT-inf-propagate", stream, kPosInf, kPosInf, 1.0f, 5) != 0) {
        failed++;
    }
    // 用例 5：start=nan -> 全 nan（nan + i*step = nan）
    if (RunFloatCaseExplicitN("case5-FLOAT-nan-propagate", stream, kNaN, 0.0f, 1.0f, 5) != 0) {
        failed++;
    }

    // 3. 释放 device 资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failed == 0) {
        LOG_PRINT("ALL EXAMPLE CASES PASS\n");
        return 0;
    }
    LOG_PRINT("EXAMPLE FAILED: %d case(s) failed\n", failed);
    return 1;
}
