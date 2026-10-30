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
 * @file test_aclnn_inv_grad.cpp
 * @brief InvGrad 算子调用示例（支持 float32 / float16 / bfloat16）
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>

#include "acl/acl.h"
#include "aclnn_inv_grad.h"

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
// Golden: dx = -dy * y * y
// ============================================================================

static float goldenInvGrad(float y, float dy) { return -dy * y * y; }

// ============================================================================
// Precision comparison (rtol/atol style)
// ============================================================================

static bool compareResult(float npuOut, float expected, float atol, float rtol)
{
    if (std::isnan(npuOut) && std::isnan(expected))
        return true;
    if (std::isinf(npuOut) && std::isinf(expected) && npuOut == expected)
        return true;
    return std::fabs(npuOut - expected) <= atol + rtol * std::fabs(expected);
}

// ============================================================================
// Per-dtype test runner
// ============================================================================

static constexpr int64_t ROWS = 2;
static constexpr int64_t COLS = 4;
static constexpr int64_t ELEM_COUNT = ROWS * COLS;

// y = forward output of Inv (y = 1/x)
static const float INPUT_Y[ELEM_COUNT] = {1.0f, 0.5f, 0.25f, -0.5f, 2.0f, -1.0f, 4.0f, -0.25f};

// dy = upstream gradient
static const float INPUT_DY[ELEM_COUNT] = {1.0f, 1.0f, 1.0f, 1.0f, 0.5f, -1.0f, 0.1f, 2.0f};

// ---------- float32 ----------

static int runFp32(aclrtStream stream)
{
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    const size_t dataBytes = ELEM_COUNT * sizeof(float);

    void *devY = nullptr, *devDy = nullptr, *devDx = nullptr, *ws = nullptr;
    aclTensor *yT = nullptr, *dyT = nullptr, *dxT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devY, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDy, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDx, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devDx, dataBytes, 0, dataBytes));
    CHECK_ACL(aclrtMemcpy(devY, dataBytes, INPUT_Y, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devDy, dataBytes, INPUT_DY, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    yT = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devY);
    dyT = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devDy);
    dxT = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devDx);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGradGetWorkspaceSize(yT, dyT, dxT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInvGrad(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    {
        float hostDx[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostDx, dataBytes, devDx, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "\n[float32] InvGrad Example (shape: [2,4])" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %11s | %9s\n", "Idx", "y", "dy", "NPU dx", "Expected", "Diff");
        std::cout << "-------------------------------------------------------------------------" << std::endl;

        int pass = 0;
        for (int i = 0; i < ELEM_COUNT; ++i) {
            float expected = goldenInvGrad(INPUT_Y[i], INPUT_DY[i]);
            float diff = std::fabs(hostDx[i] - expected);
            bool ok = compareResult(hostDx[i], expected, 1e-4f, 1e-4f);
            pass += ok ? 1 : 0;
            printf(
                "  [%d,%d] | %11.5f | %11.5f | %11.5f | %11.5f | %9.2e %s\n", (int)(i / COLS), (int)(i % COLS),
                INPUT_Y[i], INPUT_DY[i], hostDx[i], expected, diff, ok ? "PASS" : "FAIL");
        }
        std::cout << "Result: " << pass << "/" << ELEM_COUNT << (pass == ELEM_COUNT ? " -- ALL PASS" : " -- FAILED")
                  << std::endl;
        ret = (pass == ELEM_COUNT) ? 0 : 1;
    }

cleanup:
    if (yT)
        aclDestroyTensor(yT);
    if (dyT)
        aclDestroyTensor(dyT);
    if (dxT)
        aclDestroyTensor(dxT);
    if (ws)
        aclrtFree(ws);
    if (devY)
        aclrtFree(devY);
    if (devDy)
        aclrtFree(devDy);
    if (devDx)
        aclrtFree(devDx);
    return ret;
}

// ---------- float16 ----------

static int runFp16(aclrtStream stream)
{
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    const size_t dataBytes = ELEM_COUNT * sizeof(uint16_t);

    uint16_t hostY[ELEM_COUNT], hostDy[ELEM_COUNT];
    for (int i = 0; i < ELEM_COUNT; ++i) {
        hostY[i] = floatToFp16(INPUT_Y[i]);
        hostDy[i] = floatToFp16(INPUT_DY[i]);
    }

    void *devY = nullptr, *devDy = nullptr, *devDx = nullptr, *ws = nullptr;
    aclTensor *yT = nullptr, *dyT = nullptr, *dxT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devY, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDy, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDx, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devDx, dataBytes, 0, dataBytes));
    CHECK_ACL(aclrtMemcpy(devY, dataBytes, hostY, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devDy, dataBytes, hostDy, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    yT = aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, shape, 2, devY);
    dyT = aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, shape, 2, devDy);
    dxT = aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, shape, 2, devDx);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGradGetWorkspaceSize(yT, dyT, dxT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInvGrad(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    {
        uint16_t hostDx[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostDx, dataBytes, devDx, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "\n[float16] InvGrad Example (shape: [2,4])" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %11s | %9s\n", "Idx", "y", "dy", "NPU dx", "Expected", "Diff");
        std::cout << "-------------------------------------------------------------------------" << std::endl;

        int pass = 0;
        for (int i = 0; i < ELEM_COUNT; ++i) {
            float yVal = fp16ToFloat(hostY[i]);
            float dyVal = fp16ToFloat(hostDy[i]);
            float npuOut = fp16ToFloat(hostDx[i]);
            float expected = goldenInvGrad(yVal, dyVal);
            float diff = std::fabs(npuOut - expected);
            bool ok = compareResult(npuOut, expected, 1e-3f, 1e-3f);
            pass += ok ? 1 : 0;
            printf(
                "  [%d,%d] | %11.5f | %11.5f | %11.5f | %11.5f | %9.2e %s\n", (int)(i / COLS), (int)(i % COLS), yVal,
                dyVal, npuOut, expected, diff, ok ? "PASS" : "FAIL");
        }
        std::cout << "Result: " << pass << "/" << ELEM_COUNT << (pass == ELEM_COUNT ? " -- ALL PASS" : " -- FAILED")
                  << std::endl;
        ret = (pass == ELEM_COUNT) ? 0 : 1;
    }

cleanup:
    if (yT)
        aclDestroyTensor(yT);
    if (dyT)
        aclDestroyTensor(dyT);
    if (dxT)
        aclDestroyTensor(dxT);
    if (ws)
        aclrtFree(ws);
    if (devY)
        aclrtFree(devY);
    if (devDy)
        aclrtFree(devDy);
    if (devDx)
        aclrtFree(devDx);
    return ret;
}

// ---------- bfloat16 ----------

static int runBf16(aclrtStream stream)
{
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    const size_t dataBytes = ELEM_COUNT * sizeof(uint16_t);

    uint16_t hostY[ELEM_COUNT], hostDy[ELEM_COUNT];
    for (int i = 0; i < ELEM_COUNT; ++i) {
        hostY[i] = floatToBf16(INPUT_Y[i]);
        hostDy[i] = floatToBf16(INPUT_DY[i]);
    }

    void *devY = nullptr, *devDy = nullptr, *devDx = nullptr, *ws = nullptr;
    aclTensor *yT = nullptr, *dyT = nullptr, *dxT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devY, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDy, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDx, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devDx, dataBytes, 0, dataBytes));
    CHECK_ACL(aclrtMemcpy(devY, dataBytes, hostY, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devDy, dataBytes, hostDy, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    yT = aclCreateTensor(shape, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, shape, 2, devY);
    dyT = aclCreateTensor(shape, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, shape, 2, devDy);
    dxT = aclCreateTensor(shape, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, shape, 2, devDx);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGradGetWorkspaceSize(yT, dyT, dxT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInvGrad(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    {
        uint16_t hostDx[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostDx, dataBytes, devDx, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "\n[bfloat16] InvGrad Example (shape: [2,4])" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %11s | %9s\n", "Idx", "y", "dy", "NPU dx", "Expected", "Diff");
        std::cout << "-------------------------------------------------------------------------" << std::endl;

        int pass = 0;
        for (int i = 0; i < ELEM_COUNT; ++i) {
            float yVal = bf16ToFloat(hostY[i]);
            float dyVal = bf16ToFloat(hostDy[i]);
            float npuOut = bf16ToFloat(hostDx[i]);
            float expected = goldenInvGrad(yVal, dyVal);
            float diff = std::fabs(npuOut - expected);
            bool ok = compareResult(npuOut, expected, 1e-3f, 1e-3f);
            pass += ok ? 1 : 0;
            printf(
                "  [%d,%d] | %11.5f | %11.5f | %11.5f | %11.5f | %9.2e %s\n", (int)(i / COLS), (int)(i % COLS), yVal,
                dyVal, npuOut, expected, diff, ok ? "PASS" : "FAIL");
        }
        std::cout << "Result: " << pass << "/" << ELEM_COUNT << (pass == ELEM_COUNT ? " -- ALL PASS" : " -- FAILED")
                  << std::endl;
        ret = (pass == ELEM_COUNT) ? 0 : 1;
    }

cleanup:
    if (yT)
        aclDestroyTensor(yT);
    if (dyT)
        aclDestroyTensor(dyT);
    if (dxT)
        aclDestroyTensor(dxT);
    if (ws)
        aclrtFree(ws);
    if (devY)
        aclrtFree(devY);
    if (devDy)
        aclrtFree(devDy);
    if (devDx)
        aclrtFree(devDx);
    return ret;
}

// ---------- empty tensor (shape=[0]) ----------

static int runEmptyTensor(aclrtStream stream)
{
    const int64_t shape[] = {0};
    const int64_t strides[] = {1};
    const size_t dataBytes = 0;

    void *devY = nullptr, *devDy = nullptr, *devDx = nullptr, *ws = nullptr;
    aclTensor *yT = nullptr, *dyT = nullptr, *dxT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devY, 1, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDy, 1, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDx, 1, ACL_MEM_MALLOC_HUGE_FIRST));

    yT = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 1, devY);
    dyT = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 1, devDy);
    dxT = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 1, devDx);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGradGetWorkspaceSize(yT, dyT, dxT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInvGrad(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    std::cout << "\n[empty tensor] InvGrad Example (shape: [0]) -- PASS (no crash)" << std::endl;
    ret = 0;

cleanup:
    if (yT)
        aclDestroyTensor(yT);
    if (dyT)
        aclDestroyTensor(dyT);
    if (dxT)
        aclDestroyTensor(dxT);
    if (ws)
        aclrtFree(ws);
    if (devY)
        aclrtFree(devY);
    if (devDy)
        aclrtFree(devDy);
    if (devDx)
        aclrtFree(devDx);
    return ret;
}

// ---------- non-contiguous tensor (stride != shape) ----------

static int runNonContiguous(aclrtStream stream)
{
    // Non-contiguous inputs: logical shape [2,2] stored in [2,4] buffer with strides [4,1]
    // Input elements at storage offsets 0,1,4,5
    // Output tensor is contiguous [2,2] with strides [2,1] — framework contiguifies inputs
    const int64_t shape[] = {2, 2};
    const int64_t inStorageShape[] = {2, 4};
    const int64_t inStrides[] = {4, 1};
    const int64_t outStrides[] = {2, 1};
    const int64_t elemCount = 4;
    const size_t inStorageBytes = 8 * sizeof(float);
    const size_t outBytes = elemCount * sizeof(float);

    float hostY[8] = {1.0f, 0.5f, 0.0f, 0.0f, 2.0f, -1.0f, 0.0f, 0.0f};
    float hostDy[8] = {1.0f, 1.0f, 0.0f, 0.0f, 0.5f, -1.0f, 0.0f, 0.0f};
    // Logical values: y=[1.0, 0.5, 2.0, -1.0], dy=[1.0, 1.0, 0.5, -1.0]
    const float expectedY[4] = {1.0f, 0.5f, 2.0f, -1.0f};
    const float expectedDy[4] = {1.0f, 1.0f, 0.5f, -1.0f};

    void *devY = nullptr, *devDy = nullptr, *devDx = nullptr, *ws = nullptr;
    aclTensor *yT = nullptr, *dyT = nullptr, *dxT = nullptr;
    int ret = 1;

    CHECK_ACL(aclrtMalloc(&devY, inStorageBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDy, inStorageBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devDx, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devDx, outBytes, 0, outBytes));
    CHECK_ACL(aclrtMemcpy(devY, inStorageBytes, hostY, inStorageBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devDy, inStorageBytes, hostDy, inStorageBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    yT = aclCreateTensor(shape, 2, ACL_FLOAT, inStrides, 0, ACL_FORMAT_ND, inStorageShape, 2, devY);
    dyT = aclCreateTensor(shape, 2, ACL_FLOAT, inStrides, 0, ACL_FORMAT_ND, inStorageShape, 2, devDy);
    dxT = aclCreateTensor(shape, 2, ACL_FLOAT, outStrides, 0, ACL_FORMAT_ND, shape, 2, devDx);

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        CHECK_ACL(aclnnInvGradGetWorkspaceSize(yT, dyT, dxT, &wsSize, &exec));
        if (wsSize > 0)
            CHECK_ACL(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclnnInvGrad(ws, wsSize, exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    {
        float hostDx[4] = {};
        CHECK_ACL(aclrtMemcpy(hostDx, outBytes, devDx, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "\n[non-contiguous] InvGrad Example (shape: [2,2], input strides: [4,1])" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %11s | %9s\n", "Idx", "y", "dy", "NPU dx", "Expected", "Diff");
        std::cout << "-------------------------------------------------------------------------" << std::endl;

        int pass = 0;
        const int logicalIdx[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        for (int i = 0; i < elemCount; ++i) {
            int r = logicalIdx[i][0];
            int c = logicalIdx[i][1];
            float yVal = expectedY[i];
            float dyVal = expectedDy[i];
            float npuOut = hostDx[i];
            float expected = goldenInvGrad(yVal, dyVal);
            float diff = std::fabs(npuOut - expected);
            bool ok = compareResult(npuOut, expected, 1e-4f, 1e-4f);
            pass += ok ? 1 : 0;
            printf(
                "  [%d,%d] | %11.5f | %11.5f | %11.5f | %11.5f | %9.2e %s\n", r, c, yVal, dyVal, npuOut, expected, diff,
                ok ? "PASS" : "FAIL");
        }
        std::cout << "Result: " << pass << "/" << elemCount << (pass == elemCount ? " -- ALL PASS" : " -- FAILED")
                  << std::endl;
        ret = (pass == elemCount) ? 0 : 1;
    }

cleanup:
    if (yT)
        aclDestroyTensor(yT);
    if (dyT)
        aclDestroyTensor(dyT);
    if (dxT)
        aclDestroyTensor(dxT);
    if (ws)
        aclrtFree(ws);
    if (devY)
        aclrtFree(devY);
    if (devDy)
        aclrtFree(devDy);
    if (devDx)
        aclrtFree(devDx);
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
    failures += runEmptyTensor(stream);
    failures += runNonContiguous(stream);

    std::cout << "\n=========================================================================" << std::endl;
    if (failures == 0) {
        std::cout << "ALL DTYPES PASS" << std::endl;
    } else {
        std::cout << failures << " dtype(s) FAILED" << std::endl;
    }
    std::cout << "=========================================================================" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return failures;
}
