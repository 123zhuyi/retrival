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
 * \file test_reduce_std_with_mean.cpp
 * \brief Kernel unit tests for ReduceStdWithMean operator
 *
 * Coverage targets:
 *   - fp16: small 1D, 2D, 3D shapes
 *   - invert=true / false
 *   - correction=0 / 1
 *   - Different reduce lengths
 *
 * TilingKey mapping (REDUCE_STD_SCH_*):
 *   REDUCE_STD_SCH_FP16 = 0, REDUCE_STD_SCH_FP32 = 1, REDUCE_STD_SCH_BF16 = 2
 */

#include "../../../op_kernel/reduce_std_with_mean.cpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {

template <typename T1, typename T2>
T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
T CastInput(float value)
{
    return static_cast<T>(value);
}

template <typename T>
float ToFloat(T value)
{
    return static_cast<float>(value);
}

std::vector<float> ReduceStdWithMeanGolden(const std::vector<float>& self, const std::vector<float>& mean,
                                           int64_t nonReduce, int64_t reduceLen, int64_t correction, float eps,
                                           bool invert)
{
    std::vector<float> golden(static_cast<size_t>(nonReduce), 0.0f);
    for (int64_t m = 0; m < nonReduce; ++m) {
        float sumSq = 0.0f;
        for (int64_t i = 0; i < reduceLen; ++i) {
            float diff = self[m * reduceLen + i] - mean[m * reduceLen + i];
            sumSq += diff * diff;
        }
        float denom = static_cast<float>(reduceLen - correction);
        if (denom < 0.0f) {
            denom = 0.0f;
        }
        float var = (denom > 0.0f) ? (sumSq / denom) : 0.0f;
        if (invert) {
            float tmp = std::sqrt(var + eps);
            golden[m] = (tmp > 0.0f) ? (1.0f / tmp) : 0.0f;
        } else {
            golden[m] = std::sqrt(var);
        }
    }
    return golden;
}

template <typename T, uint32_t schMode>
void RunReduceStdWithMeanCase(const std::vector<float>& selfData, const std::vector<float>& meanData, int64_t nonReduce,
                              int64_t reduceLen, int64_t correction, float eps, bool invert, float atol, float rtol)
{
    const size_t inputCount = static_cast<size_t>(nonReduce * reduceLen);
    const size_t outputCount = static_cast<size_t>(nonReduce);
    const size_t inputByteSize = inputCount * sizeof(T);
    const size_t outputByteSize = outputCount * sizeof(T);

    std::vector<T> typedSelf(inputCount);
    std::vector<T> typedMean(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
        typedSelf[i] = CastInput<T>(selfData[i]);
        typedMean[i] = CastInput<T>(meanData[i]);
    }

    uint8_t* selfGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize, 32)));
    uint8_t* meanGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize, 32)));
    uint8_t* outputGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ReduceStdWithMeanTilingData)));
    ASSERT_NE(selfGM, nullptr);
    ASSERT_NE(meanGM, nullptr);
    ASSERT_NE(outputGM, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memcpy(selfGM, typedSelf.data(), inputByteSize);
    std::memcpy(meanGM, typedMean.data(), inputByteSize);
    std::memset(outputGM, 0, outputByteSize);

    // Populate TilingData
    auto* tilingData = reinterpret_cast<ReduceStdWithMeanTilingData*>(tiling);
    tilingData->totalNonReduce = nonReduce;
    tilingData->reduceLength = reduceLen;
    tilingData->blockFactor = nonReduce;
    tilingData->ubLength = reduceLen;
    tilingData->correction = correction;
    tilingData->eps = eps;
    tilingData->invert = invert;

    uint32_t blockDim = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = reduce_std_with_mean<schMode>;
    ICPU_RUN_KF(func, blockDim, selfGM, meanGM, outputGM, workspace, tiling);

    const auto* output = reinterpret_cast<const T*>(outputGM);
    const std::vector<float> golden = ReduceStdWithMeanGolden(selfData, meanData, nonReduce, reduceLen, correction, eps,
                                                              invert);
    for (size_t i = 0; i < outputCount; ++i) {
        const float actual = ToFloat(output[i]);
        const float expect = golden[i];
        EXPECT_NEAR(actual, expect, atol + rtol * std::abs(expect))
            << "Mismatch at index " << i << ": actual=" << actual << ", expect=" << expect;
    }

    AscendC::GmFree(selfGM);
    AscendC::GmFree(meanGM);
    AscendC::GmFree(outputGM);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Helper to generate float test data
std::vector<float> MakeRandomData(size_t count, float low, float high)
{
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i] = low + r * (high - low);
    }
    return data;
}

class ReduceStdWithMeanKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceStdWithMeanKernelTest SetUp" << std::endl;
        srand(42);
    }
    static void TearDownTestCase() { std::cout << "ReduceStdWithMeanKernelTest TearDown" << std::endl; }
};

} // namespace

// ==========================================================================
// fp16 (DTYPE_SELF=half, schMode=REDUCE_STD_SCH_FP16=0)
// ==========================================================================

TEST_F(ReduceStdWithMeanKernelTest, fp16_1d_small_std)
{
    const int64_t nonReduce = 1;
    const int64_t reduceLen = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 2e-2f, 2e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp16_2d_std)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -3.0f, 3.0f);
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 2e-2f, 2e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp16_3d_std)
{
    const int64_t nonReduce = 6; // 2 * 3
    const int64_t reduceLen = 16;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 2e-2f, 2e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp16_invert_true)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 0, 0.001f, true, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp16_correction_1)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -3.0f, 3.0f);
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 1, 0.0f, false, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp16_invert_true_correction_1)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 1, 0.001f, true, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp16_large_reduce_dim)
{
    const int64_t nonReduce = 2;
    const int64_t reduceLen = 128;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    // Large reduce dim means more UB tiles; use relaxed tolerance
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 5e-2f, 5e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp16_correction_0_zero_mean)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = std::vector<float>(nonReduce * reduceLen, 0.0f);
    RunReduceStdWithMeanCase<half, 0>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 2e-2f, 2e-2f);
}

// ==========================================================================
// fp32 (DTYPE_SELF=float, schMode=REDUCE_STD_SCH_FP32=1)
// Build with: --kernel_template_input='DTYPE_SELF=float'
// ==========================================================================
#ifdef REDUCE_STD_KERNEL_TEST_FP32

TEST_F(ReduceStdWithMeanKernelTest, fp32_2d_std)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -3.0f, 3.0f);
    RunReduceStdWithMeanCase<float, 1>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp32_invert_true)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 64;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    RunReduceStdWithMeanCase<float, 1>(selfData, meanData, nonReduce, reduceLen, 0, 0.001f, true, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp32_correction_1)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -3.0f, 3.0f);
    RunReduceStdWithMeanCase<float, 1>(selfData, meanData, nonReduce, reduceLen, 1, 0.0f, false, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanKernelTest, fp32_large_reduce)
{
    const int64_t nonReduce = 2;
    const int64_t reduceLen = 256;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    RunReduceStdWithMeanCase<float, 1>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 1e-5f, 1e-5f);
}

#endif // REDUCE_STD_KERNEL_TEST_FP32

// ==========================================================================
// bf16 (DTYPE_SELF=bfloat, schMode=REDUCE_STD_SCH_BF16=2)
// Build with: --kernel_template_input='DTYPE_SELF=bfloat'
// ==========================================================================
#ifdef REDUCE_STD_KERNEL_TEST_BF16

TEST_F(ReduceStdWithMeanKernelTest, bf16_2d_std)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -3.0f, 3.0f);
    RunReduceStdWithMeanCase<bfloat, 2>(selfData, meanData, nonReduce, reduceLen, 0, 0.0f, false, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, bf16_invert_true)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -2.0f, 2.0f);
    RunReduceStdWithMeanCase<bfloat, 2>(selfData, meanData, nonReduce, reduceLen, 0, 0.001f, true, 4e-2f, 4e-2f);
}

TEST_F(ReduceStdWithMeanKernelTest, bf16_correction_1)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    auto meanData = MakeRandomData(nonReduce * reduceLen, -3.0f, 3.0f);
    RunReduceStdWithMeanCase<bfloat, 2>(selfData, meanData, nonReduce, reduceLen, 1, 0.0f, false, 4e-2f, 4e-2f);
}
#endif // REDUCE_STD_KERNEL_TEST_BF16
