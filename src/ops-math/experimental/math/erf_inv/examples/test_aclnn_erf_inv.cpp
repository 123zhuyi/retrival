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
 * \file test_aclnn_erf_inv.cpp
 * \brief Comprehensive aclnn test for ErfInv operator.
 *
 * Test coverage:
 *   - Shapes: small (7), medium (1024), large (100000), multi-dim (32x32), non-aligned (3x5)
 *   - Dtypes: float32, float16
 *   - Values: zero, small (±0.1), medium (±0.5), large (±0.8), boundary (±0.999)
 *
 * Verification: roundtrip erf(erfinv(x)) == x
 *
 * Usage:
 *   export ASCEND_RT_VISIBLE_DEVICES=7
 *   source opp/vendors/custom_math/bin/set_env.bash
 *   ./test_aclnn_erf_inv
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn_erf_inv.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CHECK_ACL(expr, msg)                                        \
    do {                                                            \
        auto _ret = (expr);                                         \
        if (_ret != ACL_SUCCESS) {                                  \
            printf("[FAIL] %s — ACL error %d\n", (msg), (int)_ret); \
            return -1;                                              \
        }                                                           \
    } while (0)

static int64_t ShapeSize(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s)
        n *= d;
    return n;
}

// Float16 helpers (IEEE 754 half-precision)
static uint16_t FloatToHalf(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0)
        return sign;
    if (exp >= 31)
        return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            float r;
            uint32_t b = sign;
            std::memcpy(&r, &b, 4);
            return r;
        }
        // Subnormal
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        uint32_t b = sign | 0x7F800000 | (mant << 13);
        float r;
        std::memcpy(&r, &b, 4);
        return r;
    }
    uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

// ---------------------------------------------------------------------------
// Single test case runner
// ---------------------------------------------------------------------------

struct TestCase {
    std::string name;
    std::vector<int64_t> shape;
    aclDataType dtype;
    float atol; // absolute tolerance for roundtrip
};

static int RunTestCase(aclrtStream stream, const TestCase& tc, const std::vector<float>& values)
{
    int64_t n = ShapeSize(tc.shape);
    bool isFp16 = (tc.dtype == ACL_FLOAT16);
    size_t elemSize = isFp16 ? 2 : 4;
    size_t totalBytes = n * elemSize;

    // Prepare host data
    std::vector<uint8_t> xHostBytes(totalBytes);
    std::vector<float> xFloat(n); // keep float copy for verification

    for (int64_t i = 0; i < n; i++) {
        float v = values[i % values.size()];
        xFloat[i] = v;
        if (isFp16) {
            uint16_t h = FloatToHalf(v);
            std::memcpy(xHostBytes.data() + i * 2, &h, 2);
        } else {
            std::memcpy(xHostBytes.data() + i * 4, &v, 4);
        }
    }

    std::vector<uint8_t> outHostBytes(totalBytes, 0);

    // Allocate device memory
    void *xDev = nullptr, *outDev = nullptr;
    CHECK_ACL(aclrtMalloc(&xDev, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    CHECK_ACL(aclrtMalloc(&outDev, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc out");
    CHECK_ACL(aclrtMemcpy(xDev, totalBytes, xHostBytes.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D x");
    CHECK_ACL(aclrtMemcpy(outDev, totalBytes, outHostBytes.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D out");

    // Create tensors
    std::vector<int64_t> strides(tc.shape.size(), 1);
    for (int64_t i = (int64_t)tc.shape.size() - 2; i >= 0; i--)
        strides[i] = tc.shape[i + 1] * strides[i + 1];

    aclTensor* xT = aclCreateTensor(
        tc.shape.data(), tc.shape.size(), tc.dtype, strides.data(), 0, ACL_FORMAT_ND, tc.shape.data(), tc.shape.size(),
        xDev);
    aclTensor* outT = aclCreateTensor(
        tc.shape.data(), tc.shape.size(), tc.dtype, strides.data(), 0, ACL_FORMAT_ND, tc.shape.data(), tc.shape.size(),
        outDev);

    // Two-stage aclnn call
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnErfInvGetWorkspaceSize(xT, outT, &wsSize, &executor), "GetWorkspaceSize");

    void* wsAddr = nullptr;
    if (wsSize > 0) {
        CHECK_ACL(aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");
    }

    CHECK_ACL(aclnnErfInv(wsAddr, wsSize, executor, stream), "ErfInv execute");
    CHECK_ACL(aclrtSynchronizeStream(stream), "sync");

    // Read back
    CHECK_ACL(aclrtMemcpy(outHostBytes.data(), totalBytes, outDev, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");

    // Verify: erf(erfinv(x)) should equal x
    int failCount = 0;
    float maxDiff = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float npuVal;
        if (isFp16) {
            uint16_t h;
            std::memcpy(&h, outHostBytes.data() + i * 2, 2);
            npuVal = HalfToFloat(h);
        } else {
            std::memcpy(&npuVal, outHostBytes.data() + i * 4, 4);
        }
        float roundtrip = std::erf(npuVal);
        float diff = std::fabs(roundtrip - xFloat[i]);
        if (diff > maxDiff)
            maxDiff = diff;
        if (diff > tc.atol) {
            if (failCount < 5) { // print first 5 failures
                printf(
                    "    x=%.6f  erfinv=%.6f  erf(erfinv)=%.6f  diff=%.2e > atol=%.2e\n", xFloat[i], npuVal, roundtrip,
                    diff, tc.atol);
            }
            failCount++;
        }
    }

    // Cleanup
    aclDestroyTensor(xT);
    aclDestroyTensor(outT);
    aclrtFree(xDev);
    aclrtFree(outDev);
    if (wsSize > 0)
        aclrtFree(wsAddr);

    if (failCount > 0) {
        printf("  [FAIL] %s — %d/%ld failures, max_diff=%.2e\n", tc.name.c_str(), failCount, (long)n, maxDiff);
        return -1;
    }
    printf("  [PASS] %s — %ld elems, max_diff=%.2e\n", tc.name.c_str(), (long)n, maxDiff);
    return 0;
}

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

// Fixed values covering key regions of erfinv domain (-1, 1)
static std::vector<float> FixedValues()
{
    return {-0.999f, -0.99f, -0.9f, -0.8f, -0.5f, -0.3f, -0.1f, -0.01f, 0.0f,
            0.01f,   0.1f,   0.3f,  0.5f,  0.8f,  0.9f,  0.99f, 0.999f};
}

// Random values uniformly in (-0.99, 0.99) — avoid ±1 singularity
static std::vector<float> RandomValues(int64_t n, uint32_t seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.99f, 0.99f);
    std::vector<float> v(n);
    for (auto& x : v)
        x = dist(gen);
    return v;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    // Init ACL — ASCEND_RT_VISIBLE_DEVICES remaps to logical device 0
    aclrtStream stream;
    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(0), "setDevice");
    CHECK_ACL(aclrtCreateStream(&stream), "createStream");

    printf("\n===== ErfInv Comprehensive Test =====\n\n");

    int totalTests = 0, passedTests = 0;

    // --- float32 tests ---
    printf("[float32]\n");
    {
        // Fixed values — boundary + key regions
        TestCase tc{"fp32_fixed_17vals", {17}, ACL_FLOAT, 1e-4f};
        auto vals = FixedValues();
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }
    {
        // Small non-aligned shape
        TestCase tc{"fp32_shape_3x5", {3, 5}, ACL_FLOAT, 1e-4f};
        auto vals = FixedValues();
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }
    {
        // Medium — multi-tile single core
        TestCase tc{"fp32_shape_1024", {1024}, ACL_FLOAT, 1e-4f};
        auto vals = RandomValues(1024);
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }
    {
        // Medium — 2D
        TestCase tc{"fp32_shape_32x32", {32, 32}, ACL_FLOAT, 1e-4f};
        auto vals = RandomValues(32 * 32);
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }
    {
        // Large — multi-core
        TestCase tc{"fp32_shape_100000", {100000}, ACL_FLOAT, 1e-4f};
        auto vals = RandomValues(100000);
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }

    // --- float16 tests ---
    printf("\n[float16]\n");
    {
        // Fixed values — boundary + key regions
        TestCase tc{"fp16_fixed_17vals", {17}, ACL_FLOAT16, 5e-2f};
        auto vals = FixedValues();
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }
    {
        // Non-aligned shape
        TestCase tc{"fp16_shape_7", {7}, ACL_FLOAT16, 5e-2f};
        auto vals = FixedValues();
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }
    {
        // Medium
        TestCase tc{"fp16_shape_1024", {1024}, ACL_FLOAT16, 5e-2f};
        auto vals = RandomValues(1024, 123);
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }
    {
        // Large — multi-core
        TestCase tc{"fp16_shape_100000", {100000}, ACL_FLOAT16, 5e-2f};
        auto vals = RandomValues(100000, 456);
        totalTests++;
        if (RunTestCase(stream, tc, vals) == 0)
            passedTests++;
    }

    // --- Summary ---
    printf("\n===== Summary: %d/%d passed =====\n", passedTests, totalTests);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return (passedTests == totalTests) ? 0 : 1;
}
