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
 * @file test_aclnn_inv.cpp
 * @brief Inv 算子调用示例（支持 float32 / float16 / bfloat16）
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>

#include "acl/acl.h"
#include "aclnn_inv.h"

#define CHECK_ACL(expr)                                                                                          \
    do {                                                                                                         \
        auto _ret = (expr);                                                                                      \
        if (_ret != ACL_SUCCESS) {                                                                               \
            std::cerr << "ACL Error: " << #expr << " returned " << _ret << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                                              \
            goto cleanup;                                                                                        \
        }                                                                                                        \
    } while (0)

// ============================================================================
// dtype conversion helpers
// ============================================================================

static uint16_t floatToFp16(float val)
{
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    if (exp <= 0)
        return static_cast<uint16_t>(sign);
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static float fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = sign;
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

static uint16_t floatToBf16(float val)
{
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));
    uint32_t roundBit = (f >> 15) & 1;
    uint32_t stickyBits = f & 0x7FFF;
    uint32_t lsb = (f >> 16) & 1;
    uint32_t result = f >> 16;
    if (roundBit && (stickyBits || lsb)) {
        result += 1;
    }
    return static_cast<uint16_t>(result);
}

static float bf16ToFloat(uint16_t b)
{
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// ============================================================================
// Per-dtype test runner
// ============================================================================

static constexpr int64_t ROWS = 3;
static constexpr int64_t COLS = 5;
static constexpr int64_t ELEM_COUNT = ROWS * COLS;

static const float INPUT_DATA[ELEM_COUNT] = {-100.0f, -4.0f, -1.0f, -0.5f, -0.01f, 0.01f,   0.25f,  0.5f,
                                             1.0f,    2.0f,  4.0f,  10.0f, 100.0f, 1000.0f, -0.001f};

static bool compareResult(float npuOut, float expected, float atol, float rtol)
{
    if (std::isnan(npuOut) && std::isnan(expected))
        return true;
    if (std::isinf(npuOut) && std::isinf(expected) && npuOut == expected)
        return true;
    return std::fabs(npuOut - expected) <= atol + rtol * std::fabs(expected);
}

// ---------- float32 ----------

static int runFp32(aclrtStream stream)
{
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    const size_t dataBytes = ELEM_COUNT * sizeof(float);

    void *devIn = nullptr, *devOut = nullptr, *ws = nullptr;
    aclTensor *selfT = nullptr, *outT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devIn, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devOut, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devOut, dataBytes, 0, dataBytes));
    CHECK_ACL(aclrtMemcpy(devIn, dataBytes, INPUT_DATA, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    selfT = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devIn);
    outT = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devOut);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGetWorkspaceSize(selfT, outT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInv(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    {
        float hostOut[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostOut, dataBytes, devOut, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "\n[float32] Inv Example (shape: [3,5])" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %9s\n", "Idx", "Input", "NPU Output", "Expected", "Diff");
        std::cout << "---------------------------------------------------------------" << std::endl;

        int pass = 0;
        for (int i = 0; i < ELEM_COUNT; ++i) {
            float expected = 1.0f / INPUT_DATA[i];
            float diff = std::fabs(hostOut[i] - expected);
            bool ok = compareResult(hostOut[i], expected, 1e-4f, 1e-4f);
            pass += ok ? 1 : 0;
            printf(
                "  [%d,%d] | %11.5f | %11.5f | %11.5f | %9.2e %s\n", (int)(i / COLS), (int)(i % COLS), INPUT_DATA[i],
                hostOut[i], expected, diff, ok ? "PASS" : "FAIL");
        }
        std::cout << "Result: " << pass << "/" << ELEM_COUNT << (pass == ELEM_COUNT ? " -- ALL PASS" : " -- FAILED")
                  << std::endl;
        ret = (pass == ELEM_COUNT) ? 0 : 1;
    }

cleanup:
    if (selfT)
        aclDestroyTensor(selfT);
    if (outT)
        aclDestroyTensor(outT);
    if (ws)
        aclrtFree(ws);
    if (devIn)
        aclrtFree(devIn);
    if (devOut)
        aclrtFree(devOut);
    return ret;
}

// ---------- float16 ----------

static int runFp16(aclrtStream stream)
{
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    const size_t dataBytes = ELEM_COUNT * sizeof(uint16_t);

    uint16_t hostIn[ELEM_COUNT];
    for (int i = 0; i < ELEM_COUNT; ++i)
        hostIn[i] = floatToFp16(INPUT_DATA[i]);

    void *devIn = nullptr, *devOut = nullptr, *ws = nullptr;
    aclTensor *selfT = nullptr, *outT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devIn, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devOut, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devOut, dataBytes, 0, dataBytes));
    CHECK_ACL(aclrtMemcpy(devIn, dataBytes, hostIn, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    selfT = aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, shape, 2, devIn);
    outT = aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, shape, 2, devOut);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGetWorkspaceSize(selfT, outT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInv(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    {
        uint16_t hostOut[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostOut, dataBytes, devOut, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "\n[float16] Inv Example (shape: [3,5])" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %9s\n", "Idx", "Input", "NPU Output", "Expected", "Diff");
        std::cout << "---------------------------------------------------------------" << std::endl;

        int pass = 0;
        for (int i = 0; i < ELEM_COUNT; ++i) {
            float x = fp16ToFloat(hostIn[i]);
            float npuOut = fp16ToFloat(hostOut[i]);
            float expected = 1.0f / fp16ToFloat(floatToFp16(INPUT_DATA[i]));
            float diff = std::fabs(npuOut - expected);
            bool ok = compareResult(npuOut, expected, 1e-3f, 1e-3f);
            pass += ok ? 1 : 0;
            printf(
                "  [%d,%d] | %11.5f | %11.5f | %11.5f | %9.2e %s\n", (int)(i / COLS), (int)(i % COLS), x, npuOut,
                expected, diff, ok ? "PASS" : "FAIL");
        }
        std::cout << "Result: " << pass << "/" << ELEM_COUNT << (pass == ELEM_COUNT ? " -- ALL PASS" : " -- FAILED")
                  << std::endl;
        ret = (pass == ELEM_COUNT) ? 0 : 1;
    }

cleanup:
    if (selfT)
        aclDestroyTensor(selfT);
    if (outT)
        aclDestroyTensor(outT);
    if (ws)
        aclrtFree(ws);
    if (devIn)
        aclrtFree(devIn);
    if (devOut)
        aclrtFree(devOut);
    return ret;
}

// ---------- bfloat16 ----------

static int runBf16(aclrtStream stream)
{
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    const size_t dataBytes = ELEM_COUNT * sizeof(uint16_t);

    uint16_t hostIn[ELEM_COUNT];
    for (int i = 0; i < ELEM_COUNT; ++i)
        hostIn[i] = floatToBf16(INPUT_DATA[i]);

    void *devIn = nullptr, *devOut = nullptr, *ws = nullptr;
    aclTensor *selfT = nullptr, *outT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devIn, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devOut, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devOut, dataBytes, 0, dataBytes));
    CHECK_ACL(aclrtMemcpy(devIn, dataBytes, hostIn, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    selfT = aclCreateTensor(shape, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, shape, 2, devIn);
    outT = aclCreateTensor(shape, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, shape, 2, devOut);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGetWorkspaceSize(selfT, outT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInv(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    {
        uint16_t hostOut[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostOut, dataBytes, devOut, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "\n[bfloat16] Inv Example (shape: [3,5])" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %9s\n", "Idx", "Input", "NPU Output", "Expected", "Diff");
        std::cout << "---------------------------------------------------------------" << std::endl;

        int pass = 0;
        for (int i = 0; i < ELEM_COUNT; ++i) {
            float x = bf16ToFloat(hostIn[i]);
            float npuOut = bf16ToFloat(hostOut[i]);
            float expected = 1.0f / bf16ToFloat(floatToBf16(INPUT_DATA[i]));
            float diff = std::fabs(npuOut - expected);
            bool ok = compareResult(npuOut, expected, 1e-3f, 1e-3f);
            pass += ok ? 1 : 0;
            printf(
                "  [%d,%d] | %11.5f | %11.5f | %11.5f | %9.2e %s\n", (int)(i / COLS), (int)(i % COLS), x, npuOut,
                expected, diff, ok ? "PASS" : "FAIL");
        }
        std::cout << "Result: " << pass << "/" << ELEM_COUNT << (pass == ELEM_COUNT ? " -- ALL PASS" : " -- FAILED")
                  << std::endl;
        ret = (pass == ELEM_COUNT) ? 0 : 1;
    }

cleanup:
    if (selfT)
        aclDestroyTensor(selfT);
    if (outT)
        aclDestroyTensor(outT);
    if (ws)
        aclrtFree(ws);
    if (devIn)
        aclrtFree(devIn);
    if (devOut)
        aclrtFree(devOut);
    return ret;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    aclrtStream stream = nullptr;

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return 1;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl;
        return 1;
    }
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed" << std::endl;
        return 1;
    }

    int failures = 0;
    failures += runFp32(stream);
    failures += runFp16(stream);
    failures += runBf16(stream);

    std::cout << "\n===============================================================" << std::endl;
    if (failures == 0) {
        std::cout << "ALL DTYPES PASS" << std::endl;
    } else {
        std::cout << failures << " dtype(s) FAILED" << std::endl;
    }
    std::cout << "===============================================================" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return failures;
}
